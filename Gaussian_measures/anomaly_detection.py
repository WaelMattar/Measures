import Gaussian_measures.functions as uf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
import numpy as np

# setting
level_of_sampling = 5
decomposition_levels = 4
num_of_samples = 10*2**(level_of_sampling-1) + 1
curve = uf.one_dimensional_normal_curve(num_of_samples=num_of_samples)

# create anomalies
abnormal_curve = curve.copy()
abnormal_curve[70:90, 1] = 1/3 * abnormal_curve[70:90, 1]
# abnormal_curve[20:30, 1] = 1/1.03 * abnormal_curve[20:30, 1]
# abnormal_curve[130:150, 1] = 1/1.03 * abnormal_curve[130:150, 1]

# show curve
uf.show_one_dimensional_normal_curve(domain=np.linspace(-1, 2, 200), curve=abnormal_curve, alpha=1, save_as='abnormal_curve')

# pyramid transform
pyramid = uf.elementary_normal_multiscale_transform(curve=abnormal_curve, levels=decomposition_levels)
pyramid_norms = uf.elementary_normal_multiscale_transform_norms(curve=abnormal_curve, levels=decomposition_levels)

# plot
fig, axs = plt.subplots(decomposition_levels, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)
for layer, k in zip(pyramid_norms, range(decomposition_levels)):
    points = np.linspace(0, 1, len(layer))
    axs[k].scatter(points, pyramid_norms[k], c=points, cmap=cm.get_cmap('jet'), s=30, edgecolors='k')
    axs[k].set_yticks([0, np.max(pyramid_norms[k])], fontsize=25)
    axs[k].set_ylim([-np.min(pyramid_norms[k]), np.max(pyramid_norms[k])*1.1])
    axs[k].set_xticks(np.linspace(0, 1, 11), fontsize=25)
    axs[k].spines[['right', 'top']].set_visible(False)

axs[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))
# axs[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.4f}"))
plt.savefig('Figures/abnormal_pyramid.pdf', format='pdf', bbox_inches='tight')
plt.show()
