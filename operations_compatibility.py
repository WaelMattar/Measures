import numpy as np
import functions as uf
import matplotlib.pyplot as plt
import ot

plt.rcParams["font.family"] = "Times New Roman"

points_num = 20
measures_num = 5

sequence = uf.get_seq(measures_num=measures_num, points_num=points_num)

probabilities = ot.unif(points_num)

details = uf.o_minus(m_mu=probabilities, X_mu=sequence[0, :, :], m_nu=probabilities, X_nu=sequence[-1, :, :])

test = uf.o_plus(details, m_mu=probabilities, X_mu=sequence[0, :, :])

print('Reconstruction error is: {}'.format(np.linalg.norm(np.subtract(test, sequence[-1, :, :]))))
