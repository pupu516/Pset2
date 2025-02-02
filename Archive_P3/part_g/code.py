import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the lifting map
def f(x, y):
    return x**2 + y**2

# Define finite difference for partial derivatives
def finite_difference(f, x, y, h=1e-5):
    fx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    fy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
    return fx, fy, fxx, fxy, fyy

# Define the first fundamental form (metric tensor)
def first_fundamental_form(fx, fy):
    E = 1 + fx**2
    F = fx * fy
    G = 1 + fy**2
    return np.array([[E, F], [F, G]])

# Define the second fundamental form
def second_fundamental_form(fx, fy, fxx, fxy, fyy):
    denom = np.sqrt(1 + fx**2 + fy**2)
    L = fxx / denom
    M = fxy / denom
    N = fyy / denom
    return np.array([[L, M], [M, N]])

# Define the shape operator
def shape_operator(I, II):
    I_inv = np.linalg.inv(I)
    return np.dot(I_inv, II)

# Compute principal curvatures, Gaussian curvature, and mean curvature
def compute_curvatures(S):
    eigenvalues = np.linalg.eigvals(S)
    kappa1, kappa2 = eigenvalues
    K = kappa1 * kappa2  # Gaussian curvature
    H = (kappa1 + kappa2) / 2  # Mean curvature
    return kappa1, kappa2, K, H

# Create a grid of x, y points
x = np.linspace(-2, 2, 50)  # Resolution for visualization
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# Lift the 2D grid into 3D
Z = f(X, Y)

# Flatten the grid for computation
points_2d = np.vstack((X.flatten(), Y.flatten())).T
points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

# Initialize arrays to store curvatures
kappa1_values = np.zeros_like(X)
kappa2_values = np.zeros_like(X)
K_values = np.zeros_like(X)
H_values = np.zeros_like(X)

# Compute curvatures at each point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_val, y_val = X[i, j], Y[i, j]
        fx, fy, fxx, fxy, fyy = finite_difference(f, x_val, y_val)
        I = first_fundamental_form(fx, fy)
        II = second_fundamental_form(fx, fy, fxx, fxy, fyy)
        S = shape_operator(I, II)
        kappa1, kappa2, K, H = compute_curvatures(S)
        kappa1_values[i, j] = kappa1
        kappa2_values[i, j] = kappa2
        K_values[i, j] = K
        H_values[i, j] = H

# Plot the surface with curvature visualizations
fig = plt.figure(figsize=(18, 12))

# Plot Gaussian curvature
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(K_values), rstride=1, cstride=1, alpha=0.8)
ax1.set_title('Gaussian Curvature (K)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Plot mean curvature
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(H_values), rstride=1, cstride=1, alpha=0.8)
ax2.set_title('Mean Curvature (H)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

# Plot principal curvature kappa1
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(kappa1_values), rstride=1, cstride=1, alpha=0.8)
ax3.set_title('Principal Curvature (κ₁)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

# Plot principal curvature kappa2
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(kappa2_values), rstride=1, cstride=1, alpha=0.8)
ax4.set_title('Principal Curvature (κ₂)')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')

plt.tight_layout()
plt.savefig('shape_operator.png')
