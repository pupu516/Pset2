import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.colors import Normalize
from matplotlib.path import Path

# Step 1: Create a 2D mesh grid in the region [-2, 2] x [-2, 2]
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Flatten the grid for triangulation
points_2d = np.vstack((X.flatten(), Y.flatten())).T

# Step 2: Triangulate the 2D grid
tri = Delaunay(points_2d)

# Step 3: Lift the 2D mesh into 3D using z = x^2 + y^2
points_3d = np.hstack((points_2d, (points_2d[:, 0]**2 + points_2d[:, 1]**2).reshape(-1, 1)))

# Step 4: Calculate area changes for each triangle
def compute_triangle_area(points):
    """Compute the area of a triangle given its vertices."""
    a = np.linalg.norm(points[1] - points[0])
    b = np.linalg.norm(points[2] - points[1])
    c = np.linalg.norm(points[0] - points[2])
    s = (a + b + c) / 2  # Semi-perimeter
    return np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula

# Initialize an array to store area ratios
area_ratios = np.zeros(len(tri.simplices))

# Iterate over each triangle
for i, simplex in enumerate(tri.simplices):
    # Get the 2D vertices of the triangle
    vertices_2d = points_2d[simplex]
    
    # Compute the 2D area
    area_2d = compute_triangle_area(vertices_2d)
    
    # Get the 3D vertices after lifting
    vertices_3d = points_3d[simplex]
    
    # Compute the 3D area
    area_3d = compute_triangle_area(vertices_3d)
    
    # Compute the area ratio
    area_ratios[i] = area_3d / area_2d

# Step 5: Create a 2D heatmap
# Initialize the heatmap grid
heatmap = np.zeros_like(X)

# Map area ratios to the heatmap grid
for i, simplex in enumerate(tri.simplices):
    # Get the vertices of the triangle
    vertices = points_2d[simplex]
    
    # Create a mask for the triangle
    path = Path(vertices)
    grid_points = np.vstack((X.flatten(), Y.flatten())).T
    mask = path.contains_points(grid_points).reshape(X.shape)
    
    # Assign the area ratio to the triangle's region
    heatmap[mask] = area_ratios[i]

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(heatmap, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis', norm=Normalize(vmin=np.min(area_ratios), vmax=np.max(area_ratios)))
plt.colorbar(label='Area Ratio (3D / 2D)')
plt.title('Area Ratio Heatmap After Lifting')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('heatmap.png')  # Save the Delaunay triangulation plot
