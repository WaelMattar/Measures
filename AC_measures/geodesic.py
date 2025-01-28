import AC_measures.functions as uf
import numpy as np

num_of_samples = 17
times = 4
x = np.linspace(-1, 2, 200)

curve = uf.one_dimensional_normal_curve(num_of_samples=num_of_samples)
endpoints = np.vstack((curve[0, :], curve[-1, :]))
geodesic = uf.elementary_normal_refinement_multiple_times(endpoints, times=times)

# uf.show_one_dimensional_normal_curve(domain=x, curve=curve, alpha=1)
uf.show_one_dimensional_normal_curve(domain=x, curve=geodesic, alpha=1)
