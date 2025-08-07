import numpy as np
import matplotlib.pyplot as plt
import math
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

        # calculate the new distribution
        X = []
        for i in zip(A[0], A[1]):
            X.append(self.weight * X_mu[i[0], :] + (1-self.weight) * X_nu[i[1], :])

        X = np.reshape(X, (-1, 2))
        return G2, X


def get_curve(number_of_points: int = 10, number_of_samples: int = 640):

    # Sample points
    core = np.ones(shape=(number_of_points, 2))
    core[:, 0] = -2.5
    core[:, 1] = 1
    initial_points = np.random.normal(0, 0.3, number_of_points*2).reshape((number_of_points, 2))  # 0.2 for experiment 2
    initial_points = core + initial_points

    # Calculate curve
    curve = [initial_points]
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    f_x = (x + 1) / pow(((x + 1) ** 2 + y ** 2), 3/2) - (x - 1) / pow(((x - 1) ** 2 + y ** 2), 3/2)
    f_y = y / pow(((x + 1) ** 2 + y ** 2), 3/2) - y / pow(((x - 1) ** 2 + y ** 2), 3/2)
    delta_t = 0.15
    noise_epsilon = 0.1  # 0.08 for noisy curve
    for k in range(number_of_samples):
        new_cloud = []
        for point_num in range(number_of_points):
            new_x = f_x.subs({x: curve[-1][point_num, 0], y: curve[-1][point_num, 1]})
            new_y = f_y.subs({x: curve[-1][point_num, 0], y: curve[-1][point_num, 1]})
            if noise_epsilon > 0:
                new_x += np.random.normal(0, noise_epsilon)
                new_y += np.random.normal(0, noise_epsilon)
            new_point = np.array([curve[-1][point_num, 0] + delta_t * new_x, curve[-1][point_num, 1] + delta_t * new_y], dtype=np.float64)
            new_cloud.append(new_point)
        curve.append(np.vstack(new_cloud))
    return curve


def downsampling(curve: list):
    down = [curve[2*k] for k in range(int(len(curve)/2)+1)]
    return down


def linear_interpolation_refinement(curve: list):
    P_half = P_weight(weight=0.5)
    L = len(curve) - 1
    new_curve = [curve[0]]
    m_mu = np.ones(curve[0].shape[0]) / curve[0].shape[0]
    m_nu = np.ones(curve[0].shape[0]) / curve[0].shape[0]
    for i in range(L):
        _, new_cloud = P_half(m_mu=m_mu, X_mu=curve[i], m_nu=m_nu, X_nu=curve[i+1])
        new_curve.append(new_cloud)
        new_curve.append(curve[i+1])
    return new_curve


def linear_interpolation_refinement_multiple_times(curve: np.ndarray, times: int):
    refined_curve = curve.copy()
    for _ in range(times):
        refined_curve = linear_interpolation_refinement(curve=refined_curve)
    return refined_curve


def o_minus(a: np.ndarray, b: np.ndarray):  # a ominus b
    m_mu = np.ones(a.shape[0]) / a.shape[0]
    m_nu = np.ones(b.shape[0]) / b.shape[0]
    M2 = ot.dist(a, b, metric='sqeuclidean')
    M2 /= M2.max()
    G2 = ot.emd(m_mu, m_nu, M2)
    A = np.matrix.nonzero(G2)
    res = []
    for k, j in zip(A[0], A[1]):
        res.append([k, j, np.subtract(b[j], a[k])])
    return res


def o_plus(b: np.ndarray, detail_coeff: list):
    a = b.copy()
    for vec in detail_coeff:
        k, j, diff = vec
        a[k] = b[j] + (-1)*diff  # pullback to a
    return a


def wasserstein_distance(a: np.ndarray, b: np.ndarray):
    m_mu = np.ones(a.shape[0]) / a.shape[0]
    m_nu = np.ones(b.shape[0]) / b.shape[0]
    M2 = ot.dist(a, b, metric='sqeuclidean')
    M2 /= M2.max()
    G2 = ot.emd(m_mu, m_nu, M2)
    return math.sqrt(np.sum(G2*M2))


def decomposition(curve: list):
    down = downsampling(curve)
    ref = linear_interpolation_refinement(curve=down)
    details = [o_minus(a=curve[k], b=ref[k]) for k in range(len(ref))]
    return [down, details]


def decomposition_norms(curve: list):
    down = downsampling(curve)
    ref = linear_interpolation_refinement(curve=down)
    details = [wasserstein_distance(a=curve[k], b=ref[k]) for k in range(len(ref))]
    return [down, details]


def elementary_multiscale_transform(curve: list, levels: int):
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
        temp = [o_plus(b=refinement[i], detail_coeff=pyramid[level+1][i]) for i in range(len(refinement))]
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
