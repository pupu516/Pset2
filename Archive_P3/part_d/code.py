import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the lifting map
def f(x, y):
    return x**2 + y**2

# Define the partial derivatives of the lifting map
def df_dx(x, y):
    return 2 * x

def df_dy(x, y):
    return 2 * y

# Define the normal of the lifted map
def lifted_map_normal(x, y):
    grad_x = df_dx(x, y)
    grad_y = df_dy(x, y)
    normal = np.array([-grad_x, -grad_y, 1])
    normal /= np.linalg.norm(normal)  # Normalize to unit vector
    return normal

# Create a grid of x, y points
x = np.linspace(-2, 2, 20)  # Reduced resolution for better visualization
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Lift the 2D grid into 3D
Z = f(X, Y)

# Flatten the grid for triangulation
points_2d = np.vstack((X.flatten(), Y.flatten())).T
points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

# Perform Delaunay triangulation
tri = Delaunay(points_2d)

# Compute the normal of the lifted map at each point
normals = np.zeros_like(points_3d)
for i, (x_val, y_val, z_val) in enumerate(points_3d):
    normals[i] = lifted_map_normal(x_val, y_val)

# Plot the lifted mesh and normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, cmap='viridis', alpha=0.8)

# Plot the normals as quivers
ax.quiver(
    points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],  # Start points of the arrows
    normals[:, 0], normals[:, 1], normals[:, 2],        # Direction of the arrows
    length=0.2, color='r', normalize=True               # Length and color of the arrows
)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lifted Mesh with Normals of the Lifted Map')
plt.savefig('liftmap_with_normal.png')
