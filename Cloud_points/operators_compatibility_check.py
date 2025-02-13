import numpy as np
import functions as uf
np.random.seed(5)

a = np.random.normal(loc=0, scale=1, size=200).reshape(-1, 2)
b = np.random.normal(loc=0, scale=1, size=200).reshape(-1, 2)

# a = np.array([[0, 2], [0, 3]], dtype=float).reshape(-1, 2)
# b = np.array([[2, 0], [3, 0]], dtype=float).reshape(-1, 2)

dif = uf.o_minus(a=a, b=b)
a_ = uf.o_plus(b, dif)

print('Error = {}'.format(np.linalg.norm(a-a_)))
