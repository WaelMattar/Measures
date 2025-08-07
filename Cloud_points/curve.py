import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import functions as uf
np.random.seed(5)

# 1D arrays
x = np.arange(-7, 3, 0.1)
y = np.arange(-1, 9, 0.1)

# Meshgrid
X, Y = np.meshgrid(x, y)

# Assign vector directions
Ex = (X + 1) / pow(((X + 1) ** 2 + Y ** 2), 3/2) - (X - 1) / pow(((X - 1) ** 2 + Y ** 2), 3/2)
Ey = Y / pow(((X + 1) ** 2 + Y ** 2), 3/2) - Y / pow(((X - 1) ** 2 + Y ** 2), 3/2)

curve = uf.get_curve(number_of_points=10)

# Depict illustration
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Ex, Ey, density=1, linewidth=.3, color='k')  # color='#A23BEC'
plt.scatter(-1, 0, c='r', s=60)
plt.scatter(1, 0, c='b', s=60)
for point_num in range(10):
    single_point_x = [curve[k][point_num, 0] for k in range(len(curve))]
    single_point_y = [curve[k][point_num, 1] for k in range(len(curve))]
    color_scheme = np.linspace(0, 1, len(curve))
    plt.scatter(single_point_x, single_point_y, c=color_scheme, cmap=cm.get_cmap('jet'), s=10)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

# Show plot
plt.gca().set_aspect('equal')
plt.savefig('Figures/curve_electromagnetic.pdf', format='pdf', bbox_inches='tight')
plt.show()
