import Gaussian_measures.functions as uf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rc('font', family='Times New Roman')

# setting
level_of_sampling = 6
decomposition_levels = 6
num_of_samples = 10*2**(level_of_sampling-1) + 1
endpoints = uf.one_dimensional_normal_curve(num_of_samples=2)
geodesic = uf.elementary_normal_refinement_multiple_times(curve=endpoints, times=decomposition_levels+1)
curve = uf.one_dimensional_normal_curve(num_of_samples=len(geodesic))

# calculating Delta(curve)
diff = [uf.normal_wasserstein_distance(measure_0=curve[i, :], measure_1=curve[i+1, :]) for i in range(curve.shape[0]-1)]
delta = np.max(diff)
Gamma = delta/(1/(num_of_samples-1))

# plot max norms
plt.figure(num=2, figsize=(8, 6))
for k in np.linspace(0, 1, 5):
    average = k*curve + (1-k)*geodesic
    # uf.show_one_dimensional_normal_curve(domain=np.linspace(-1, 2, 200), curve=average, alpha=1, save_as='average_'+str(k))
    pyramid_norms = uf.elementary_normal_multiscale_transform_norms(curve=average, levels=decomposition_levels)
    error = [np.max(np.abs(pyramid_norms[i])) for i in range(decomposition_levels)]
    error = [error[i] if error[i] != 0 else 2**(-30) for i in range(len(error))]
    plt.plot(np.linspace(1, decomposition_levels, decomposition_levels), np.log2(error), linewidth=3, label=r'k = {}'.format(k))
    plt.scatter(np.linspace(1, decomposition_levels, decomposition_levels), np.log2(error))

# plot theoretical bound
# plt.plot(np.linspace(1, decomposition_levels, decomposition_levels),
#          np.log2(Gamma)+(1-(np.linspace(1, decomposition_levels, decomposition_levels))), linewidth=3, label=r'Theoretical bound', linestyle='dashed', color='k')
plt.legend(loc='best', fontsize='14')
plt.xticks(np.linspace(1, decomposition_levels, decomposition_levels), fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Figures/details_decay.pdf', format='pdf', bbox_inches='tight')
plt.show()

# average plotting and print optimality
level_of_sampling = 6
decomposition_levels = 6
num_of_samples = 10*2**(level_of_sampling-1) + 1
endpoints = uf.one_dimensional_normal_curve(num_of_samples=2)
geodesic = uf.elementary_normal_refinement_multiple_times(curve=endpoints, times=decomposition_levels+1)
curve = uf.one_dimensional_normal_curve(num_of_samples=len(geodesic))
for k in np.linspace(0, 1, 5):
    average = k*curve + (1-k)*geodesic
    print('For k = {} the optimality number is {}.'.format(k, uf.elementary_curve_optimality(curve=average, levels=decomposition_levels)))
    uf.show_one_dimensional_normal_curve(domain=np.linspace(-1, 2, 200), curve=average, alpha=1, save_as='average_'+str(k))
