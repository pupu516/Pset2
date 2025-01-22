import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def local_ortho_coor(theta, phi):
    er1 = np.sin(theta) * np.cos(phi)
    er2 = np.sin(theta) * np.sin(phi)
    er3 = np.cos(theta)
    et1 = np.cos(theta) * np.cos(phi)
    et2 = np.cos(theta) * np.sin(phi)
    et3 = -np.sin(theta)
    ep1 = -np.sin(phi)
    ep2 = np.cos(phi)
    ep3 = 0

    e_r = np.array([er1, er2, er3])
    e_theta = np.array([et1, et2, et3])
    e_phi = np.array([ep1, ep2, ep3])

    return e_r, e_theta, e_phi

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.sin(v), np.cos(u))
y = np.outer(np.sin(v), np.sin(u))
z = np.outer(np.cos(v), np.ones_like(u))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.6)

density_theta = 10
density_phi = 15
theta_example = np.linspace(0, np.pi, density_theta)
phi_example = np.linspace(0, 2 * np.pi, density_phi)
theta_grid, phi_grid = np.meshgrid(theta_example, phi_example)

for i in range(theta_grid.shape[0]):
    for j in range(theta_grid.shape[1]):
        theta = theta_grid[i, j]
        phi = phi_grid[i, j]

        e_r, e_theta, e_phi = local_ortho_coor(theta, phi)

        origin = np.array([np.sin(theta) * np.cos(phi),
                           np.sin(theta) * np.sin(phi),
                           np.cos(theta)])

        ax.quiver(*origin, *e_r, color='r', length=0.2, normalize=True)
        ax.quiver(*origin, *e_theta, color='g', length=0.2, normalize=True)
        ax.quiver(*origin, *e_phi, color='b', length=0.2, normalize=True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=30)

plt.savefig("unit_sphere_with_dense_coordinates.png", dpi=300, bbox_inches='tight')

