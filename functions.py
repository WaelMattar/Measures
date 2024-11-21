import numpy as np
import matplotlib.pyplot as plt
import ot

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['image.composite_image'] = False
np.random.seed(5)


class P_a:
    def __init__(self, a: float):
        self.a = a

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


def get_seq(measures_num: int = 512, points_num: int = 100):
    sequence = np.zeros(shape=(measures_num, points_num, 2))

    for i in range(measures_num):
        mean = [0, 0]  # [0, 10*np.exp(-3*i/measures_num)]
        cov = np.identity(2)  # *(10*np.exp(-7*i/measures_num))
        sequence[i, :, :] = np.random.multivariate_normal(mean, cov, points_num)

    # sorting each measure with respect to the first coordinate
    z = []
    for i in range(measures_num):
        y = sequence[i]
        y.sort(axis=0)
        z.append(y)
    return np.stack(z)  # np.array(measures_num, points_num, dimension)


def get_seq_2(measures_num: int = 512, points_num: int = 100):
    sequence = np.zeros(shape=(measures_num, points_num, 2))

    for i in range(points_num):
        sequence[:, i, 0] = np.linspace(-np.pi, np.pi, measures_num) + 0.0005*(i+1)

    sequence[:, 0, 1] = np.cos(sequence[:, 0, 0])
    sequence[:, 1, 1] = np.pi - np.abs(sequence[:, 1, 0])
    sequence[:, 2, 1] = 0.6*np.sin(sequence[:, 2, 0]) + 0.5*np.cos(2*sequence[:, 2, 0]) - 0.25*np.sin(4*sequence[:, 2, 0]) - 2

    # sorting each measure with respect to the first coordinate
    z = []
    for i in range(measures_num):
        y = sequence[i]
        y.sort(axis=0)
        z.append(y)
    return np.stack(z)  # np.array(measures_num, points_num, dimension)


def show_seq(sequence: np.ndarray, increments: float = 0):
    plt.figure(figsize=(8, 6), num=999)
    for i in range(sequence.shape[0]):
        plt.scatter(sequence[i, :, 0] + increments*i, sequence[i, :, 1])

    plt.show()


def downsample(sequence: np.ndarray):
    return np.delete(sequence, obj=[2*k+1 for k in range(0, int(sequence.shape[0]/2))], axis=0)


def linear_interpolation_refinement(sequence: np.ndarray, probabilities: np.ndarray):
    P_half = P_a(a=0.5)
    refined_sequence = []
    A = []
    G = []

    for i in range(sequence.shape[0]):
        refined_sequence.append(sequence[i])
        G.append(probabilities[i])

        if i == sequence.shape[0]-1:
            break

        g, a, refined = P_half(m_mu=probabilities[i], X_mu=sequence[i], m_nu=probabilities[i+1], X_nu=sequence[i+1])
        g = np.sum(g, axis=0)  # may change axis to 1
        refined_sequence.append(refined)
        A.append(a)
        G.append(g)

    refined_sequence = np.stack(refined_sequence)
    G = np.stack(G)
    return [G, A, refined_sequence]


def o_minus(m_mu, X_mu, m_nu, X_nu):
    P_half = P_a(a=0.5)
    G2, A, X = P_half(m_mu=m_mu, X_mu=X_mu, m_nu=m_nu, X_nu=X_nu)
    res = []

    for k, j in zip(A[0], A[1]):
        res.append([k, j, np.subtract(X_mu[k], X_nu[j])])

    return res


def o_plus(details, m_mu, X_mu):
    res = X_mu.copy()
    for det_coeff in details:
        res[det_coeff[0]] = res[det_coeff[0]] + (-1)*det_coeff[2]  # pullback

    res.sort(axis=0)  # sort with respect to the first coordinate
    return res


def pyramid_decomposition(sequence: np.ndarray, probabilities: np.ndarray, layers: int):
    deci = [sequence]
    ref = []
    p_deci = [probabilities]
    p_ref = []
    det = []

    [deci.append(downsample(sequence=deci[-1])) for _ in range(layers)]
    [p_deci.append(downsample(sequence=p_deci[-1])) for _ in range(layers)]
    [ref.append(linear_interpolation_refinement(sequence=deci[k], probabilities=p_deci[k])[-1]) for k in range(1, layers+1)]
    [p_ref.append(linear_interpolation_refinement(sequence=deci[k], probabilities=p_deci[k])[0]) for k in range(1, layers+1)]

    [det.append([o_minus(m_mu=p_ref[k][j], X_mu=ref[k][j], m_nu=p_deci[k][j], X_nu=deci[k][j]) for j in range(len(ref[k]))]) for k in range(layers)]

    det.append(deci[-1])
    return det


def pyramid_reconstruction(pyramid: np.ndarray, probabilities: np.ndarray, layers: int):
    res = pyramid[-1]
    p_res = probabilities

    for k in range(layers):
        ref = linear_interpolation_refinement(sequence=res, probabilities=p_res)
        p_res, res = ref[0], ref[-1]
        res = np.stack([o_plus(details=pyramid[-2-k][j], X_mu=res[j, :, :], m_mu=None) for j in range(len(pyramid[-2-k]))])

    return res
