import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define spherical to Cartesian transformation
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# Unit vectors in the spherical coordinate system
def e_phi(theta, phi):
    # Assuming the vector is in the phi direction
    return np.array([-np.sin(phi), np.cos(phi), 0])

# Set parameters
r = 1
theta_0 = np.pi / 4  # Fixed polar angle for simplicity
phi_values = np.linspace(0, 2 * np.pi, 20)

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)

# Plot the parallel transport of the vector along the path
for phi in phi_values:
    vector = e_phi(theta_0, phi)
    start_point = spherical_to_cartesian(r, theta_0, phi)
    ax.quiver(start_point[0], start_point[1], start_point[2], vector[0], vector[1], vector[2], length=3, color='r')

# Display the plot
plt.savefig("parallel_transport_phi.png", dpi=1000, bbox_inches='tight')

