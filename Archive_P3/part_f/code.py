import numpy as np

# Define the lifting map
def f(x, y):
    return x**2 + y**2

# Define the partial derivatives of the lifting map
def df_dx(x, y):
    return 2 * x

def df_dy(x, y):
    return 2 * y

# Define the vertex normal
def vertex_normal(x, y):
    grad_x = df_dx(x, y)
    grad_y = df_dy(x, y)
    normal = np.array([-grad_x, -grad_y, 1])
    normal /= np.linalg.norm(normal)  # Normalize to unit vector
    return normal

# Define the second fundamental form
def second_fundamental_form(x, y):
    # Second derivatives of the position vector r(x, y) = (x, y, f(x, y))
    d2r_dx2 = np.array([0, 0, 2])  # ∂²r/∂x²
    d2r_dxdy = np.array([0, 0, 0])  # ∂²r/∂x∂y
    d2r_dy2 = np.array([0, 0, 2])  # ∂²r/∂y²

    # Vertex normal
    n = vertex_normal(x, y)

    # Compute the components of the second fundamental form
    L = np.dot(n, d2r_dx2)
    M = np.dot(n, d2r_dxdy)
    N = np.dot(n, d2r_dy2)

    return np.array([[L, M], [M, N]])

# Example: Compute the second fundamental form at a specific point (x, y)
x, y = 1.0, 1.0  # Example point
II = second_fundamental_form(x, y)

print("Second Fundamental Form at (x, y) = (1, 1):")
print(II)
