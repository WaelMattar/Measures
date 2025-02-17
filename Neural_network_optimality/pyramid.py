import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functions as uf
import matplotlib.cm as cm
import seaborn as sns
from matplotlib import ticker

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['image.composite_image'] = False

# Load file of measures
digit = 3
epochs = 161
batch = 128
weights = 2346
decomposition_levels = 4

file_name = 'measures_for_digit_{}_with_{}_epochs_{}_batch_and_{}_weights.csv'.format(digit, epochs, batch, weights)
measures = pd.read_csv('Measures_results/'+file_name)

# heatmap
ax = sns.heatmap(measures, cmap='jet', xticklabels=np.linspace(0, 9, 10, dtype=int), linewidths=0)
plt.savefig('Figures/heat_map.pdf', format='pdf', bbox_inches='tight')

measures = np.array(measures)
pyramid = uf.elementary_multiscale_transform(curve=measures, levels=decomposition_levels)
reconstruction = uf.inverse_multiscale_transform(pyramid=pyramid)
pyramid_norms = uf.elementary_multiscale_transform_norms(curve=measures, levels=decomposition_levels)

fig, axs = plt.subplots(decomposition_levels, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)
for layer, k in zip(pyramid_norms, range(decomposition_levels)):
    points = np.linspace(1, 161, len(layer), dtype=float)
    axs[k].scatter(points, pyramid_norms[k], c=points, cmap=cm.get_cmap('jet'), s=30, edgecolors='k')
    axs[k].set_yticks([0, np.max(pyramid_norms[k])], fontsize=25)
    axs[k].set_ylim([-np.min(pyramid_norms[k]), np.max(pyramid_norms[k])*1.1])
    axs[k].set_xticks(np.linspace(1, 161, 11, dtype=int), fontsize=25)
    axs[k].spines[['right', 'top']].set_visible(False)

print('The optimality number is {}.'.format(uf.elementary_curve_optimality(curve=measures, levels=decomposition_levels)))
plt.savefig('Figures/pyramid.pdf', format='pdf', bbox_inches='tight')
plt.show()
