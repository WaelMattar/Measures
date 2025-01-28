import AC_measures.functions as uf
import numpy as np

num_of_samples = 12
x = np.linspace(-1, 2, 200)
curve = uf.one_dimensional_curve(num_of_samples=num_of_samples)
uf.show_one_dimensional_curve(domain=x, curve=curve, alpha=1)
