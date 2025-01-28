import AC_measures.functions as uf
import numpy as np

num_of_samples = 2
x = np.linspace(-1, 2, 200)
curve = uf.one_dimensional_normal_curve(num_of_samples=num_of_samples)

mean = uf.normal_mean(measure_0=(curve[0][0], curve[1][0]), measure_1=(curve[0][-1], curve[1][-1]), param=0.5)

refined_curve = uf.elementary_normal_refinement(curve)

print()
