import functions as uf
import matplotlib.pyplot as plt
import numpy as np
import ot

plt.rcParams["font.family"] = "Times New Roman"

points_num = 3
measures_num = 10*2**2+1

layers = 3
P_half = uf.P_a(a=0.5)
sequence = uf.get_seq(measures_num=measures_num, points_num=points_num)
probabilities = ot.unif(points_num)

pyramid = uf.pyramid_decomposition(sequence=sequence,
                                   probabilities=np.vstack([probabilities]*measures_num),
                                   layers=layers)

reconstructed = uf.pyramid_reconstruction(pyramid=pyramid,
                                          layers=layers,
                                          probabilities=np.vstack([probabilities]*len(pyramid[-1])))

diff = np.subtract(reconstructed, sequence)

print('Reconstruction error is: {}'.format(np.linalg.norm(diff)))

fig = plt.figure()
gs = fig.add_gridspec(layers, hspace=0.2)
axs = gs.subplots()
for layer, k in zip(pyramid[:-1], range(layers)):
    points = np.linspace(0, 10, len(layer))

    for det_coeff, point in zip(layer, points):

        for element in det_coeff:
            x_values = [point, point+element[-1][0]/10]
            y_values = [0, element[-1][1]/100]
            if np.max(element[-1]) == 0:
                axs[layers - k - 1].plot(x_values, y_values, 'ro', linestyle='-', markersize=1)
            else:
                axs[layers-k-1].plot(x_values, y_values, 'bo', linestyle='-', markersize=1, linewidth=1)

    # ax[layers-k-1].spines['top'].set_visible(False)
    # ax[layers-k-1].spines['right'].set_visible(False)
    # ax[k].set_ylabel('scale ' + str(k), fontsize=14)
    # ax[k].set_ylim(0, np.max(np.abs(pyramid[k] + pyramid_tilde[scale])) + 0.005, emit=True)
    # ax[k].set_xlim(min(uf.dyadic_grid(0, 10, resolution=scale + 1)),
    #                        max(uf.dyadic_grid(0, 10, resolution=scale + 1)), emit=True)
    # ax[k].set_yticks([np.max(np.abs(pyramid[scale]))])
    # ax[k].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax[layers-k-1].set_yticks([-0.04, 0.04])
    # ax[layers-k-1].set_ylim(-0.04, 0.04, emit=True)
    # axs[layers-k-1].tick_params(axis='both', which='major', labelsize=16)

plt.savefig('pyramid.pdf', format='pdf', bbox_inches='tight')
plt.show()
