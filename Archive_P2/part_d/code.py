import numpy as np
import matplotlib.pyplot as plt

# Stereographic projection
def stereographic_projection(x, y, z):
    X = x / (1 - z)
    Y = y / (1 - z)
    return X, Y

# Inverse stereographic projection
def inverse_stereographic_projection(X, Y):
    denom = 1 + X**2 + Y**2
    x = 2 * X / denom
    y = 2 * Y / denom
    z = (-1 + X**2 + Y**2) / denom
    return x, y, z

# Inner product on the sphere
def inner_product_sphere(u, v):
    return np.dot(u, v)

# Inner product on the plane
def inner_product_plane(u, v):
    return np.dot(u, v)

# Example tangent vectors at a point on the sphere
u = np.array([1, 0, 0])  # Tangent vector 1
v = np.array([0, 1, 0])  # Tangent vector 2

# Generate points on the sphere
theta = np.linspace(0, np.pi, 100)  # Polar angle
phi = np.linspace(0, 2 * np.pi, 100)  # Azimuthal angle
theta, phi = np.meshgrid(theta, phi)
x = np.sin(theta) * np.cos(phi)  # x-coordinate on sphere
y = np.sin(theta) * np.sin(phi)  # y-coordinate on sphere
z = np.cos(theta)  # z-coordinate on sphere

# Compute inner products
inner_sphere = inner_product_sphere(u, v)  # Inner product on sphere
inner_plane = np.zeros_like(x)  # Initialize inner product on plane

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        # Project the point to the plane
        X, Y = stereographic_projection(x[i, j], y[i, j], z[i, j])
        
        # Project the tangent vectors to the plane
        u_proj = np.array([X, Y])
        v_proj = np.array([-Y, X])  # Rotate 90 degrees for orthogonal vector
        
        # Compute inner product on the plane
        inner_plane[i, j] = inner_product_plane(u_proj, v_proj)

# Plot the ratio of inner products
ratio = inner_plane / inner_sphere
plt.figure(figsize=(8, 6))
plt.contourf(np.degrees(phi), np.degrees(theta), ratio, levels=50, cmap='viridis')
plt.colorbar(label='Inner Product Ratio (Plane / Sphere)')
plt.xlabel('Phi (degrees)')
plt.ylabel('Theta (degrees)')
plt.title('Ratio of Inner Products After Stereographic Projection')
plt.show()
