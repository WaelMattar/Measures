from Cloud_points import functions
import matplotlib.pyplot as plt
import ot.plot
import numpy as np
np.random.seed(1337)

mu_s = np.random.multivariate_normal(mean=[0, 0], cov=[[0.2, 0], [0, 0.2]], size=60)
mu_0 = np.random.multivariate_normal(mean=[1, 3], cov=[[0.1, 0], [0, 0.1]], size=30)
mu_t = np.random.multivariate_normal(mean=[3, 2], cov=[[0.0001, 0], [0, 0.2]], size=20)

P_half = functions.P_weight(weight=0.5)
G2, X = P_half(m_mu=np.ones(shape=60)/60, X_mu=mu_s, m_nu=np.ones(shape=20)/20, X_nu=mu_t)

plt.figure(1, figsize=(8, 6))
plt.plot(mu_s[:, 0], mu_s[:, 1], 'o', markersize=8, color='r', label='$\mu_s$ - 60 points')
# ot.plot.plot2D_samples_mat(mu_s, mu_t, G2, color=[0.5, 0.5, 0.5])
plt.plot(mu_t[:, 0], mu_t[:, 1], 'o', markersize=8, color='b', label='$\mu_t$ - 30 points')
# plt.plot(mu_0[:, 0], mu_0[:, 1], 'o', markersize=8, color='purple', label='$\mu_2$ - 20 points')
plt.plot(X[:, 0], X[:, 1], '*', markersize=8, color='purple', label='$M(\mu_s, \mu_t; 1/2)$')
plt.xlim(-1.8, 3.2)
plt.ylim(-1.5, 3.7)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)
plt.title('The average measure', fontsize=20)
plt.savefig('ST_mean_2.pdf', bbox_inches='tight', format='pdf')
plt.show()
