import numpy as np
import matplotlib.pyplot as plt
import sympy
import ot

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['image.composite_image'] = False
np.random.seed(5)


class P_weight:
    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, m_mu, X_mu, m_nu, X_nu):

        # calculating the cost matrix
        M2 = ot.dist(X_mu, X_nu, metric='sqeuclidean')
        M2 /= M2.max()

        # calculating the coupling matrix
        G2 = ot.emd(m_mu, m_nu, M2)

        # get indices of nonzero entries
        A = np.matrix.nonzero(G2)
        A = [list(A[0]), list(A[1])]

        # calculate the new density
        X = []
        for i in zip(A[0], A[1]):
            X.append(self.a * X_mu[i[0], :] + (1-self.a) * X_nu[i[1], :])

        X = np.reshape(X, (-1, 2))
        return G2, A, X  # return coupling matrix, indices of nonzero entries, and new density points


def get_curve(number_of_points: int = 10, number_of_samples: int = 160):

    # Sample points
    core = np.ones(shape=(number_of_points, 2))
    core[:, 0] = -2.5
    core[:, 1] = 1.2
    initial_points = np.random.normal(0, 0.3, 20).reshape((number_of_points, 2))
    initial_points = core + initial_points

    # Calculate curve
    curve = [initial_points]
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    f_x = (x + 1) / ((x + 1) ** 2 + y ** 2) - (x - 1) / ((x - 1) ** 2 + y ** 2)
    f_y = y / ((x + 1) ** 2 + y ** 2) - y / ((x - 1) ** 2 + y ** 2)
    delta_t = 0.2
    for k in range(number_of_samples):
        new_cloud = []
        for point_num in range(number_of_points):
            new_x = f_x.subs({x: curve[-1][point_num, 0], y: curve[-1][point_num, 1]})
            new_y = f_y.subs({x: curve[-1][point_num, 0], y: curve[-1][point_num, 1]})
            new_point = np.array([curve[-1][point_num, 0] + delta_t * new_x, curve[-1][point_num, 1] + delta_t * new_y], dtype=np.float64)
            new_cloud.append(new_point)
        curve.append(np.vstack(new_cloud))
    return curve


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


def o_minus(a: np.ndarray, b: np.ndarray):  # a ominus b
    X = np.linspace(0, 9, 10, dtype=int).tolist()
    if np.abs(np.sum(a) - np.sum(b)) >= 10 ** (-6):
        a = a / np.sum(a, dtype=np.float64)
        b = b / np.sum(b, dtype=np.float64)
    G2 = ot.emd_1d(X, X, a=a.tolist(), b=b.tolist(), metric='sqeuclidean')
    return G2 - np.diag(v=b)


def o_plus(b: np.array, detail_coeff: np.ndarray):
    coupling = detail_coeff + np.diag(v=b)
    return np.sum(coupling, axis=1)


def wasserstein_distance(a: np.ndarray, b: np.ndarray):
    X = np.linspace(0, 9, 10, dtype=int).tolist()
    if np.abs(np.sum(a) - np.sum(b)) >= 10 ** (-6):
        a = a / np.sum(a, dtype=np.float64)
        b = b / np.sum(b, dtype=np.float64)
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


def inverse_multiscale_transform(pyramid: list):
    levels = len(pyramid)
    temp = pyramid[0]
    for level in range(levels-1):
        refinement = linear_interpolation_refinement(curve=temp)
        temp = np.array([o_plus(b=refinement[i], detail_coeff=pyramid[level+1][i]) for i in range(len(refinement))])
    return temp


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
    sums = [np.sum(norm_pyramid[i]) for i in range(len(norm_pyramid))]
    return np.sum(sums)


def curve_decay_rate(curve: np.ndarray, levels: int):
    norm_pyramid = elementary_multiscale_transform_norms(curve=curve, levels=levels)
    maximums = np.array([np.max(norm_pyramid[i]) for i in range(len(norm_pyramid))])
    ratios = maximums[:-1] / maximums[1:]
    return np.mean(ratios)
