import Gaussian_measures.functions as uf
import matplotlib.pyplot as plt
import numpy as np

num_of_refinements = 4
num_of_samples = 2**num_of_refinements + 1
x = np.linspace(-1, 2, 200)

curve = uf.one_dimensional_normal_curve(num_of_samples=num_of_samples)
endpoints = np.vstack((curve[0, :], curve[-1, :]))
geodesic = uf.elementary_normal_refinement_multiple_times(endpoints, times=num_of_refinements)

uf.show_one_dimensional_normal_curve(domain=x, curve=geodesic, alpha=1)
