import functions as uf
import numpy as np

a = np.random.random(size=10)
b = np.random.random(size=10)

a = a/np.sum(a)
b = b/np.sum(b)

detail_coeff = uf.o_minus(a=a, b=b)
a_ = uf.o_plus(b=b, detail_coeff=detail_coeff)

print('Error = {}'.format(np.linalg.norm(a-a_)))
