import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the lifting map
def f(x, y):
    return x**2 + y**2

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

# Compute face normals
face_normals = np.zeros((len(tri.simplices), 3))
for i, simplex in enumerate(tri.simplices):
    # Get the vertices of the triangle
    v0, v1, v2 = points_3d[simplex]
    
    # Compute two edges of the triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Compute the face normal using the cross product
    face_normals[i] = np.cross(edge1, edge2)
    face_normals[i] /= np.linalg.norm(face_normals[i])  # Normalize

# Compute vertex normals
vertex_normals = np.zeros_like(points_3d)
for i, vertex in enumerate(points_3d):
    # Find all triangles that share this vertex
    connected_triangles = np.where(tri.simplices == i)[0]
    
    # Average the face normals of the connected triangles
    vertex_normals[i] = np.mean(face_normals[connected_triangles], axis=0)
    vertex_normals[i] /= np.linalg.norm(vertex_normals[i])  # Normalize

# Plot the lifted mesh and vertex normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, cmap='viridis', alpha=0.8)

# Plot the vertex normals as quivers
ax.quiver(
    points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],  # Start points of the arrows
    vertex_normals[:, 0], vertex_normals[:, 1], vertex_normals[:, 2],  # Direction of the arrows
    length=0.2, color='r', normalize=True               # Length and color of the arrows
)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lifted Mesh with Vertex Normals')

plt.savefig('vertex_normal.png')
