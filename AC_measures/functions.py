import numpy as np
import matplotlib.pyplot as plt
import ot

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['image.composite_image'] = False
np.random.seed(5)


def one_dimensional_normal_curve(num_of_samples: int):
    sigmas = np.linspace(0, 1, num=num_of_samples)
    sigmas = 0.4 - 1 * (sigmas - 0.46) ** 2
    means = np.linspace(0, 1, num=num_of_samples)
    return np.transpose(np.vstack((means, sigmas)))


def show_one_dimensional_normal_curve(domain: np.ndarray, curve: np.ndarray, alpha: float):
    means, sigmas = curve[:, 0], curve[:, 1]
    plt.figure(num=0, figsize=(10, 6))
    for i in range(len(means)):
        y = 1 / (2 * np.pi * sigmas[i] ** 2) * np.exp(-((domain - means[i]) ** 2) / (2 * sigmas[i] ** 2))
        plt.plot(domain, y, color=(i/len(means), 0, 1 - i/len(means)), linewidth=3, alpha=alpha)
    plt.xticks(size=16)
    plt.yticks([])
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
