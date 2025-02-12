import Gaussian_measures.functions as uf
import numpy as np

level_of_sampling = 5
decomposition_levels = 4
num_of_samples = 10*2**(level_of_sampling-1) + 1
curve = uf.one_dimensional_normal_curve(num_of_samples=num_of_samples)

measure_0 = curve[0, :]
measure_1 = curve[-1, :]

f = uf.normal_o_minus(measure_0, measure_1)
measure_0_ = uf.normal_o_plus(measure_1, f)

print('Error = {}'.format(np.linalg.norm(measure_0 - measure_0_)))
