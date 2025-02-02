import numpy as np
import matplotlib.pyplot as plt

# Define the lifting map and its partial derivatives
def f(x, y):
    return x**2 + y**2

def df_dx(x, y):
    return 2 * x

def df_dy(x, y):
    return 2 * y

# Define the induced metric analytically
def induced_metric_analytic(x, y):
    g_xx = 1 + df_dx(x, y)**2
    g_xy = df_dx(x, y) * df_dy(x, y)
    g_yy = 1 + df_dy(x, y)**2
    return np.array([[g_xx, g_xy], [g_xy, g_yy]])

# Create a grid of x, y points
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Initialize arrays to store the metric components
g_xx = np.zeros_like(X)
g_xy = np.zeros_like(X)
g_yy = np.zeros_like(X)

# Compute the metric at each point in the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        g = induced_metric_analytic(X[i, j], Y[i, j])
        g_xx[i, j] = g[0, 0]
        g_xy[i, j] = g[0, 1]
        g_yy[i, j] = g[1, 1]

# Print the metric at specific points for verification
points_to_print = [(0, 0), (1, 1), (-1, 2)]  # Example points
for (x_val, y_val) in points_to_print:
    g = induced_metric_analytic(x_val, y_val)
    print(f"Analytic Induced Metric at (x, y) = ({x_val}, {y_val}):")
    print(g)
    print()

# Plot the components of the metric as heatmaps
plt.figure(figsize=(15, 5))

# Plot g_xx
plt.subplot(1, 3, 1)
plt.imshow(g_xx, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
plt.colorbar(label='g_xx')
plt.title('g_xx: Metric Component')
plt.xlabel('x')
plt.ylabel('y')

# Plot g_xy
plt.subplot(1, 3, 2)
plt.imshow(g_xy, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
plt.colorbar(label='g_xy')
plt.title('g_xy: Metric Component')
plt.xlabel('x')
plt.ylabel('y')

# Plot g_yy
plt.subplot(1, 3, 3)
plt.imshow(g_yy, extent=(-2, 2, -2, 2), origin='lower', cmap='viridis')
plt.colorbar(label='g_yy')
plt.title('g_yy: Metric Component')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig('metric_element graph.png')  # Save the Delaunay triangulation plot
