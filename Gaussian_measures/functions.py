import numpy as np
import matplotlib.pyplot as plt
import sympy

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['image.composite_image'] = False
np.random.seed(5)


def one_dimensional_normal_curve(num_of_samples: int):
    sigmas = np.linspace(0, 1, num=num_of_samples)
    sigmas = 0.4 - 1 * (sigmas - 0.46) ** 2
    means = np.linspace(0, 1, num=num_of_samples)
    return np.transpose(np.vstack((means, sigmas)))


def one_dimensional_normal_curve_noise(num_of_samples: int):
    sigmas = np.linspace(0, 1, num=num_of_samples)
    sigmas = 0.4 - 1 * (sigmas - 0.46) ** 2
    means = np.linspace(0, 1, num=num_of_samples)
    noise = np.random.normal(0, 1, num_of_samples)
    regularized_mean_noise = np.array([noise[i]*(i/(num_of_samples-1))*(1-i/(num_of_samples-1)) for i in range(num_of_samples)])
    regularized_sigma_noise = np.array([noise[i]*np.exp(-10*np.abs(i/(num_of_samples-1)-0.5)) for i in range(num_of_samples)])
    means = means + 0.2*regularized_mean_noise
    sigmas = sigmas + 0.02*regularized_sigma_noise
    return np.transpose(np.vstack((means, sigmas)))


def show_one_dimensional_normal_curve(domain: np.ndarray, curve: np.ndarray, alpha: float, save_as: str):
    means, sigmas = curve[:, 0], curve[:, 1]
    plt.figure(num=0, figsize=(8, 6))
    N = len(means)
    colors = plt.cm.jet(np.linspace(0, 1, N))
    for i in range(len(means)):
        y = 1 / (2 * np.pi * sigmas[i] ** 2) * np.exp(-((domain - means[i]) ** 2) / (2 * sigmas[i] ** 2))
        plt.plot(domain, y, color=colors[i], linewidth=2, alpha=alpha)
    plt.xticks(size=16)
    plt.yticks([])
    plt.savefig('Figures/'+save_as+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def normal_mean(measure_0: np.ndarray, measure_1: np.ndarray, param: float):
    mean_0, sigma_0 = measure_0[0], measure_0[1]
    mean_1, sigma_1 = measure_1[0], measure_1[1]
    new_mean = (1-param)*mean_0 + param*mean_1
    C = np.sqrt(sigma_1/sigma_0)
    new_sigma = ((1-param)+param*C)*sigma_0*((1-param)+param*C)
    return new_mean, new_sigma


def elementary_normal_refinement(curve: np.ndarray):
    L = curve.shape[0] - 1
    new_curve = curve[0, :]
    for i in range(L):
        new_mean, new_sigma = normal_mean(measure_0=curve[i, :], measure_1=curve[i+1, :], param=0.5)
        refined_point = np.array([new_mean, new_sigma])
        new_curve = np.vstack((new_curve, refined_point))
        new_curve = np.vstack((new_curve, curve[i+1, :]))
    return new_curve


def elementary_normal_refinement_multiple_times(curve: np.ndarray, times: int):
    refined_curve = curve.copy()
    for _ in range(times):
        refined_curve = elementary_normal_refinement(curve=refined_curve)
    return refined_curve


def downsampling(curve: np.ndarray):
    new_curve = curve.copy()
    return np.delete(new_curve, obj=[2*k+1 for k in range(int(new_curve.shape[0]/2))], axis=0)


def normal_o_minus(measure_0: np.ndarray, measure_1: np.ndarray):
    mean_0, sigma_0 = measure_0[0], measure_0[1]
    mean_1, sigma_1 = measure_1[0], measure_1[1]
    x = sympy.Symbol('x')
    f = mean_1 + np.sqrt(sigma_1/sigma_0)*(x - mean_0) - x  # f.subs('x', 0)
    return f


def normal_wasserstein_distance(measure_0: np.ndarray, measure_1: np.ndarray):
    mean_0, sigma_0 = measure_0[0], measure_0[1]
    mean_1, sigma_1 = measure_1[0], measure_1[1]
    return np.abs(mean_0 - mean_1) + sigma_0 + sigma_1 - 2*np.sqrt(sigma_0*sigma_1)


def decomposition(curve: np.ndarray):
    down = downsampling(curve)
    ref = elementary_normal_refinement(curve=down)
    details = [normal_o_minus(measure_0=curve[k, :], measure_1=ref[k, :]) for k in range(ref.shape[0])]
    return [down, details]


def decomposition_norms(curve: np.ndarray):
    down = downsampling(curve)
    ref = elementary_normal_refinement(curve=down)
    details = [normal_wasserstein_distance(measure_0=curve[k, :], measure_1=ref[k, :]) for k in range(ref.shape[0])]
    return [down, details]


def elementary_normal_multiscale_transform(curve: np.ndarray, levels: int):
    pyramid = []
    coarse = curve.copy()
    for _ in range(levels):
        temporary = decomposition(curve=coarse)
        coarse = temporary[0]
        details = temporary[1]
        pyramid.insert(0, details)
    pyramid.insert(0, coarse)
    return pyramid


def elementary_normal_multiscale_transform_norms(curve: np.ndarray, levels: int):
    pyramid = []
    coarse = curve.copy()
    for _ in range(levels):
        temporary = decomposition_norms(curve=coarse)
        coarse = temporary[0]
        details = temporary[1]
        pyramid.insert(0, details)
    return pyramid


def elementary_curve_optimality(curve: np.ndarray, levels: int):
    norm_pyramid = elementary_normal_multiscale_transform_norms(curve=curve, levels=levels)
    sums = [np.sum(norm_pyramid[i])*2**i for i in range(len(norm_pyramid))]
    return np.sum(sums)
