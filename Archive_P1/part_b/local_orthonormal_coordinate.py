import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate local orthonormal coordinate system
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

# Plotting the 3D sphere with example local coordinates
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Generate a unit sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.sin(v), np.cos(u))
y = np.outer(np.sin(v), np.sin(u))
z = np.outer(np.cos(v), np.ones_like(u))
ax.plot_surface(x, y, z, color='lightblue', alpha=0.6)

# Example points (theta, phi)
example_points = [
    (np.pi / 4, np.pi / 4),
    (np.pi / 3, np.pi / 6),
    (np.pi / 6, np.pi / 2),
]

# Add local coordinate systems
for theta, phi in example_points:
    # Get the local orthonormal vectors
    e_r, e_theta, e_phi = local_ortho_coor(theta, phi)
    # Convert the point on the sphere (theta, phi) to Cartesian
    origin = np.array([np.sin(theta) * np.cos(phi),
                       np.sin(theta) * np.sin(phi),
                       np.cos(theta)])
    # Plot the vectors
    ax.quiver(*origin, *e_r, color='r', label='e_r' if theta == example_points[0][0] else "", length=0.2, normalize=True)
    ax.quiver(*origin, *e_theta, color='g', label='e_theta' if theta == example_points[0][0] else "", length=0.2, normalize=True)
    ax.quiver(*origin, *e_phi, color='b', label='e_phi' if theta == example_points[0][0] else "", length=0.2, normalize=True)

# Add overlay text and visual elements
fig.text(0.5, 0.95, "Audio Hijack Is Almost Ready", fontsize=16, color='orange', ha='center', fontweight='bold')
fig.text(0.5, 0.88, (
    "You now need to adjust your system's settings to allow ACE to run. "
    "In the Mac's \"Recovery\" environment, you'll modify the System Security Policy to match the settings pictured below."
), fontsize=10, color='black', ha='center', wrap=True)
fig.text(0.5, 0.02, (
    "To access the \"Recovery\" environment, shut down your Mac, then press and hold "
    "the Power button until \"Loading startup options\" appears."
), fontsize=10, color='black', ha='center', wrap=True)

# Set labels and view angle
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=20, azim=30)
ax.legend()

plt.savefig("unit_sphere_with_overlay.png", dpi=300, bbox_inches='tight')


