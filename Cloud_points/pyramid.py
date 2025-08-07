import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import functions as uf
np.random.seed(5)

curve = uf.get_curve(number_of_points=10)
decomposition_levels = 6
pyramid = uf.elementary_multiscale_transform(curve=curve, levels=decomposition_levels)
pyramid_norms = uf.elementary_multiscale_transform_norms(curve=curve, levels=decomposition_levels)
reconstruction = uf.inverse_multiscale_transform(pyramid=pyramid)

# plot coarse approximation
x = np.arange(-7, 3, 0.1)
y = np.arange(-1, 9, 0.1)

# Meshgrid
X, Y = np.meshgrid(x, y)

# Assign vector directions
Ex = (X + 1) / pow(((X + 1) ** 2 + Y ** 2), 3/2) - (X - 1) / pow(((X - 1) ** 2 + Y ** 2), 3/2)
Ey = Y / pow(((X + 1) ** 2 + Y ** 2), 3/2) - Y / pow(((X - 1) ** 2 + Y ** 2), 3/2)

# Depict illustration
coarse = pyramid[0]
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Ex, Ey, density=1, linewidth=.3, color='k')  # color='#A23BEC'
plt.scatter(-1, 0, c='r', s=60)
plt.scatter(1, 0, c='b', s=60)
for point_num in range(10):
    single_point_x = [coarse[k][point_num, 0] for k in range(len(coarse))]
    single_point_y = [coarse[k][point_num, 1] for k in range(len(coarse))]
    color_scheme = np.linspace(0, 1, len(coarse))
    plt.scatter(single_point_x, single_point_y, c=color_scheme, cmap=cm.get_cmap('jet'), s=40)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

# Show plot
plt.gca().set_aspect('equal')
plt.savefig('Figures/coarse_curve_electromagnetic.pdf', format='pdf', bbox_inches='tight')

# Plot pyramid
fig, axs = plt.subplots(decomposition_levels, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)
for layer, k in zip(pyramid_norms, range(decomposition_levels)):
    points = np.linspace(0, 161, len(layer))
    axs[k].scatter(points, pyramid_norms[k], c=points, cmap=cm.get_cmap('jet'), s=30, edgecolors='k')
    axs[k].set_yticks([0, np.max(pyramid_norms[k])], fontsize=25)
    axs[k].set_ylim([-np.min(pyramid_norms[k]), np.max(pyramid_norms[k])*1.1])
    axs[k].set_xticks(np.linspace(0, 161, 11, dtype=int), fontsize=25)
    axs[k].spines[['right', 'top']].set_visible(False)

plt.savefig('Figures/pyramid_electromagnetic.pdf', format='pdf', bbox_inches='tight')
plt.show()
