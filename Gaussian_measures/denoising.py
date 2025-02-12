import Gaussian_measures.functions as uf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
import numpy as np

# setting
level_of_sampling = 5
decomposition_levels = 4
num_of_samples = 10*2**(level_of_sampling-1) + 1
curve = uf.one_dimensional_normal_curve_noise(num_of_samples=num_of_samples)

# pyramid transform
pyramid = uf.elementary_normal_multiscale_transform(curve=curve, levels=decomposition_levels)
pyramid_norms = uf.elementary_normal_multiscale_transform_norms(curve=curve, levels=decomposition_levels)

# thresholding
sparse_pyramid = uf.pyramid_thresholding(pyramid=pyramid, pyramid_norms=pyramid_norms, threshold=0.01)

# reconstructing
sparse_curve = uf.inverse_normal_multiscale_transform(pyramid=sparse_pyramid)

# show curve
uf.show_one_dimensional_normal_curve(domain=np.linspace(-1, 2, 200), curve=sparse_curve, alpha=1, save_as='denoised_curve')

# print maximal wasserstein distance
wasserstein_distances = [uf.normal_wasserstein_distance(measure_0=curve[k, :], measure_1=sparse_curve[k, :]) for k in range(len(curve))]
print('Error = {}'.format(np.max(wasserstein_distances)))

# plot denoised pyramid
sparse_pyramid_norms = uf.elementary_normal_multiscale_transform_norms(curve=sparse_curve, levels=decomposition_levels)
fig, axs = plt.subplots(decomposition_levels, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)
for layer, k in zip(sparse_pyramid_norms, range(decomposition_levels)):
    points = np.linspace(0, 1, len(layer))
    axs[k].scatter(points, sparse_pyramid_norms[k], c=points, cmap=cm.get_cmap('jet'), s=30, edgecolors='k')
    axs[k].set_yticks([0, np.max(sparse_pyramid_norms[k])], fontsize=25)
    axs[k].set_ylim([-np.min(sparse_pyramid_norms[k]), np.max(sparse_pyramid_norms[k])*1.1])
    axs[k].set_xticks(np.linspace(0, 1, 11), fontsize=25)
    axs[k].spines[['right', 'top']].set_visible(False)

axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))
axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))
plt.savefig('Figures/denoised_pyramid.pdf', format='pdf', bbox_inches='tight')
plt.show()
