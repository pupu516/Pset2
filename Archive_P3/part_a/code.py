import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# Load the point cloud data from the .dat file, skipping the header row
points = np.loadtxt('mesh.dat', skiprows=1)

# Verify the loaded data
print("Loaded points shape:", points.shape)
print(points)

# Step 1: Plot the Convex Hull
hull = ConvexHull(points)

# Create a new figure for the convex hull
plt.figure(figsize=(6, 6))
plt.plot(points[:, 0], points[:, 1], 'o', label='Points')  # Plot the points

# Plot the convex hull edges
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', label='Convex Hull' if simplex[0] == 0 else "")

plt.title('2D Convex Hull')
plt.legend()
plt.savefig('convex_hull.png')  # Save the convex hull plot

# Step 2: Generate and Plot the Delaunay Triangulation
tri = Delaunay(points)

# Create a new figure for the Delaunay triangulation
plt.figure(figsize=(6, 6))
plt.triplot(points[:, 0], points[:, 1], tri.simplices, 'k-', label='Delaunay Triangulation')  # Plot the triangulation
plt.plot(points[:, 0], points[:, 1], 'o', label='Points')  # Plot the points

plt.title('2D Delaunay Triangulation')
plt.legend()
plt.savefig('delaunay_triangulation.png')  # Save the Delaunay triangulation plot
