import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.ops.numpy_ops import linspace

# Sphere radius
r = 1.0

# Create a smooth parametric mesh for the sphere (lat-long grid)
n_lat = 60
n_lon = 120
u = np.linspace(0, 2 * np.pi, n_lon)
v = np.linspace(0, np.pi, n_lat)
U, V = np.meshgrid(u, v)
X = r * np.cos(U) * np.sin(V)
Y = r * np.sin(U) * np.sin(V)
Z = r * np.cos(V)


# Create near-uniform points on the sphere using a Fibonacci lattice
def fibonacci_sphere(samples=400, radius=1.0):
    pts = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_xy = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_xy
        z = np.sin(theta) * radius_xy
        pts.append((radius * x, radius * y, radius * z))
    return np.array(pts)


pts = fibonacci_sphere(1000, r)

# Plot
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface (smooth)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=0.2, antialiased=True, cmap=cm.coolwarm)

# Overlay uniform points (small dots)
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.001)

# Add points and arrow
p = np.array([0, 0, 1])
q = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
ax.scatter(p[0], p[1], p[2], s=40, color='black')
ax.scatter(q[0], q[1], q[2], s=40, color='black')

theta = np.arccos(np.dot(p, q))
d = theta/np.linalg.norm(np.subtract(q, np.dot(p, q)*p))*(np.subtract(q, np.dot(p, q)*p))
ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], color='black')
ax.text(p[0], p[1], p[2] + 0.1, "$p$", color='black', fontsize=16)
ax.text(q[0], q[1]+0.05, q[2] + 0.05, "$q$", color='black', fontsize=16)
ax.text(p[0] + 0.5*d[0], p[1] + 0.5*d[1], p[2] + 0.5*d[2] + 0.1, "$q\ominus p$", color='black', fontsize=16)

x = [np.sin((1-t)*theta)/np.sin(theta)*p[0] + np.sin(t*theta)/np.sin(theta)*q[0] for t in linspace(0, 1, 20)]
y = [np.sin((1-t)*theta)/np.sin(theta)*p[1] + np.sin(t*theta)/np.sin(theta)*q[1] for t in linspace(0, 1, 20)]
z = [np.sin((1-t)*theta)/np.sin(theta)*p[2] + np.sin(t*theta)/np.sin(theta)*q[2] for t in linspace(0, 1, 20)]
geo = np.array([x, y, z]).T
ax.scatter(geo[:, 0], geo[:, 1], geo[:, 2], s=5, color='green')

# Equal aspect ratio so the sphere looks round
try:
    ax.set_box_aspect([1, 1, 1])  # Matplotlib >= 3.3
except Exception:
    # Fallback for older versions
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Remove axes, ticks and grid
ax.set_axis_off()

# A gentle viewing angle
ax.view_init(elev=0, azim=50)

# plt.tight_layout()
plt.show()
