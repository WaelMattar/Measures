import functions as uf
import matplotlib.pyplot as plt
import ot.plot
import ot

plt.rcParams["font.family"] = "Times New Roman"

points_num = 3
measures_num = 10*2**2+1

sequence = uf.get_seq_2(measures_num=measures_num, points_num=points_num)

probabilities = ot.unif(points_num)

downsampled = uf.downsample(sequence=sequence)

P_half = uf.P_a(a=0.5)

plt.figure(1, figsize=(8, 6))
for i in range(measures_num):
    plt.scatter(sequence[i, :, 0], sequence[i, :, 1], s=30, label='measure #{}'.format(i+1), zorder=2)

    if i == measures_num-1:
        break

    G2, A, X = P_half(m_mu=probabilities, X_mu=sequence[i, :, :], m_nu=probabilities, X_nu=sequence[i+1, :, :])
    ot.plot.plot2D_samples_mat(sequence[i, :, :], sequence[i+1, :, :], G2, c=[0.8, 0.8, 0.8])

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xlim([-8.5, 8.5])
# plt.ylim([-7, 7])
ax = plt.gca()
# ax.set_aspect('equal', adjustable='box')
# plt.legend(loc='best', fontsize=10)
plt.savefig('sequence.pdf', format='pdf', bbox_inches='tight')
plt.show()
