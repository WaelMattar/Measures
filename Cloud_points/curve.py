import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sympy

np.random.seed(5)

# 1D arrays
x = np.arange(-7, 3, 0.1)
y = np.arange(-1, 9, 0.1)

# Meshgrid
X, Y = np.meshgrid(x, y)

# Assign vector directions
Ex = (X + 1) / ((X + 1) ** 2 + Y ** 2) - (X - 1) / ((X - 1) ** 2 + Y ** 2)
Ey = Y / ((X + 1) ** 2 + Y ** 2) - Y / ((X - 1) ** 2 + Y ** 2)

# Sample 10 points
core = np.ones(shape=(10, 2))
core[:, 0] = -2.5
core[:, 1] = 1.2
initial_points = np.random.normal(0, 0.3, 20).reshape((10, 2))
initial_points = core + initial_points

# Calculate curve
curve = [initial_points]
x = sympy.Symbol('x')
y = sympy.Symbol('y')
f_x = (x + 1) / ((x + 1) ** 2 + y ** 2) - (x - 1) / ((x - 1) ** 2 + y ** 2)
f_y = y / ((x + 1) ** 2 + y ** 2) - y / ((x - 1) ** 2 + y ** 2)
delta_t = 0.2
for k in range(160):
    new_cloud = []
    for point_num in range(10):
        new_x = f_x.subs({x: curve[-1][point_num, 0], y: curve[-1][point_num, 1]})
        new_y = f_y.subs({x: curve[-1][point_num, 0], y: curve[-1][point_num, 1]})
        new_point = np.array([curve[-1][point_num, 0]+delta_t*new_x, curve[-1][point_num, 1]+delta_t*new_y], dtype=np.float64)
        new_cloud.append(new_point)
    curve.append(np.vstack(new_cloud))

# Depict illustration
plt.figure(figsize=(10, 10))
plt.streamplot(X, Y, Ex, Ey, density=1, linewidth=.5, color='k')  # color='#A23BEC'
plt.scatter(initial_points[:, 0], initial_points[:, 1], c='b', s=20)
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
plt.show()
