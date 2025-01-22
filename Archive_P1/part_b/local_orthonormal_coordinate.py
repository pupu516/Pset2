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

density = 20
theta_example = np.linspace(0, np.pi, density)
phi_example   = np.linspace(0, 2*np.pi, density)


example_points = np.array([theta_example, phi_example]).T

for theta, phi in example_points:
    e_r, e_theta, e_phi = local_ortho_coor(theta, phi)
    origin = np.array([np.sin(theta) * np.cos(phi),
                       np.sin(theta) * np.sin(phi),
                       np.cos(theta)])

    ax.quiver(*origin, *e_r, color='r', label='e_r' if theta == example_points[0][0] else "", length=0.2, normalize=True)
    ax.quiver(*origin, *e_theta, color='g', label='e_theta' if theta == example_points[0][0] else "", length=0.2, normalize=True)
    ax.quiver(*origin, *e_phi, color='b', label='e_phi' if theta == example_points[0][0] else "", length=0.2, normalize=True)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=30)
ax.legend()

plt.savefig("unit_sphere_with_overlay.png", dpi=300, bbox_inches='tight')


