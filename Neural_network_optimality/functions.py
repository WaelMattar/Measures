import numpy as np
import matplotlib.pyplot as plt
import ot

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['image.composite_image'] = False
np.random.seed(5)


class P_weight:
    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, a, b):

        X = np.linspace(0, 9, 10, dtype=int).tolist()

        # calculating the coupling matrix
        G2 = ot.emd_1d(X, X, a=a.tolist(), b=b.tolist(), metric='sqeuclidean')

        # get indices of nonzero entries
        A = np.matrix.nonzero(G2)
        A = [list(A[0]), list(A[1])]

        # coupling matrix dynamics  TODO: delete later but use for oplus function
        source = a.copy()
        target = np.zeros(shape=(10,))
        for f, t in zip(A[0], A[1]):
            source[f] -= G2[f, t]
            target[t] += G2[f, t]

        # calculate the new measure  TODO: verify the average!
        new = (1-self.weight)*a + self.weight*b
        return G2, new


def downsampling(curve: np.ndarray):
    new_curve = curve.copy()
    return np.delete(new_curve, obj=[2*k+1 for k in range(int(new_curve.shape[0]/2))], axis=0)


def linear_interpolation_refinement(curve: np.ndarray):
    P_half = P_weight(weight=0.5)
    L = curve.shape[0] - 1
    new_curve = curve[0, :]
    for i in range(L):
        _, new_measure = P_half(a=curve[i, :], b=curve[i+1, :])
        new_curve = np.vstack((new_curve, new_measure))
        new_curve = np.vstack((new_curve, curve[i+1, :]))
    return new_curve


def linear_interpolation_refinement_multiple_times(curve: np.ndarray, times: int):
    refined_curve = curve.copy()
    for _ in range(times):
        refined_curve = linear_interpolation_refinement(curve=refined_curve)
    return refined_curve


def o_minus(a: np.ndarray, b: np.ndarray):  # TODO: fix the subtraction
    X = np.linspace(0, 9, 10, dtype=int).tolist()
    G2 = ot.emd_1d(X, X, a=a.tolist(), b=b.tolist(), metric='sqeuclidean')
    return G2


def o_plus(b: np.array, coupling_matrix: np.ndarray):  # TODO:finish
    return


def wasserstein_distance(a: np.ndarray, b: np.ndarray):
    X = np.linspace(0, 9, 10, dtype=int).tolist()
    return np.sqrt(ot.emd2_1d(X, X, a=a.tolist(), b=b.tolist(), metric='sqeuclidean'))


def decomposition(curve: np.ndarray):
    down = downsampling(curve)
    ref = linear_interpolation_refinement(curve=down)
    details = [o_minus(a=curve[k, :], b=ref[k, :]) for k in range(ref.shape[0])]
    return [down, details]


def decomposition_norms(curve: np.ndarray):
    down = downsampling(curve)
    ref = linear_interpolation_refinement(curve=down)
    details = [wasserstein_distance(a=curve[k, :], b=ref[k, :]) for k in range(ref.shape[0])]
    return [down, details]


def elementary_multiscale_transform(curve: np.ndarray, levels: int):
    pyramid = []
    coarse = curve.copy()
    for _ in range(levels):
        temporary = decomposition(curve=coarse)
        coarse = temporary[0]
        details = temporary[1]
        pyramid.insert(0, details)
    pyramid.insert(0, coarse)
    return pyramid


def elementary_multiscale_transform_norms(curve: np.ndarray, levels: int):
    pyramid = []
    coarse = curve.copy()
    for _ in range(levels):
        temporary = decomposition_norms(curve=coarse)
        coarse = temporary[0]
        details = temporary[1]
        pyramid.insert(0, details)
    return pyramid


def elementary_curve_optimality(curve: np.ndarray, levels: int):
    norm_pyramid = elementary_multiscale_transform_norms(curve=curve, levels=levels)
    sums = [np.sum(norm_pyramid[i])*2**(i+1) for i in range(len(norm_pyramid))]
    return np.sum(sums)
