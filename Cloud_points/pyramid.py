import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import functions as uf
np.random.seed(5)

# curve = uf.get_curve(number_of_points=10, number_of_samples=10)

a = np.random.normal(loc=0, scale=1, size=4).reshape(-1, 2)
b = np.random.normal(loc=0, scale=1, size=4).reshape(-1, 2)
dif = uf.o_minus(a=a, b=b)
a_ = uf.o_plus(b, dif)
print()
