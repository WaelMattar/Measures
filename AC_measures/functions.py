import numpy as np
import matplotlib.pyplot as plt
import ot

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['image.composite_image'] = False
np.random.seed(5)


def one_dimensional_curve(num_of_samples: int):
    sigmas = np.linspace(0, 1, num=num_of_samples)
    sigmas = 0.4 - 1 * (sigmas - 0.46) ** 2
    means = np.linspace(0, 1, num=num_of_samples)
    return means, sigmas


def show_one_dimensional_curve(domain, curve, alpha):
    means, sigmas = curve
    plt.figure(num=0, figsize=(10, 6))
    for i in range(len(means)):
        y = 1 / (2 * np.pi * sigmas[i] ** 2) * np.exp(-((domain - means[i]) ** 2) / (2 * sigmas[i] ** 2))
        plt.plot(domain, y, color=(i/len(means), 0, 1 - i/len(means)), linewidth=3, alpha=alpha)
    plt.show()
