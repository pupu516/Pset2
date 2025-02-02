import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Part a: Load the point cloud, plot the convex hull, and generate the Delaunay triangulation
def part_a():
    # Load the point cloud (replace 'mesh.dat' with your file path)
    points = np.loadtxt('mesh.dat', skiprows=1)

    # Plot the convex hull
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.title('2D Convex Hull')
    plt.show()

    # Generate the Delaunay triangulation
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.title('2D Delaunay Triangulation')
    plt.show()

    return points, tri

# Part b: Lift the 2D mesh into 3D, compute area ratios, and plot a heatmap
def part_b(points, tri):
    # Define the lifting map
    def f(x, y):
        return x**2 + x * y + y**2

    # Lift the 2D points into 3D
    points_3d = np.hstack((points, f(points[:, 0], points[:, 1]).reshape(-1, 1)))

    # Compute area ratios
    def compute_triangle_area(points):
        a = np.linalg.norm(points[1] - points[0])
        b = np.linalg.norm(points[2] - points[1])
        c = np.linalg.norm(points[0] - points[2])
        s = (a + b + c) / 2
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    area_ratios = np.zeros(len(tri.simplices))
    for i, simplex in enumerate(tri.simplices):
        vertices_2d = points[simplex]
        vertices_3d = points_3d[simplex]
        area_2d = compute_triangle_area(vertices_2d)
        area_3d = compute_triangle_area(vertices_3d)
        area_ratios[i] = area_3d / area_2d

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.tripcolor(points[:, 0], points[:, 1], tri.simplices, facecolors=area_ratios, edgecolors='k')
    plt.colorbar(label='Area Ratio (3D / 2D)')
    plt.title('Area Ratio Heatmap After Lifting')
    plt.show()

    return points_3d, tri

# Part c: Calculate the induced metric analytically
def part_c(points):
    # Define the lifting map and its partial derivatives
    def f(x, y):
        return x**2 + x * y + y**2

    def df_dx(x, y):
        return 2 * x + y

    def df_dy(x, y):
        return x + 2 * y

    # Compute the induced metric
    def induced_metric(x, y):
        fx = df_dx(x, y)
        fy = df_dy(x, y)
        return np.array([[1 + fx**2, fx * fy], [fx * fy, 1 + fy**2]])

    # Example: Compute the metric at a specific point
    x, y = points[0, 0], points[0, 1]
    metric = induced_metric(x, y)
    print(f"Induced Metric at (x, y) = ({x}, {y}):")
    print(metric)

# Part d: Calculate the surface normal of the lifted mesh and plot it
def part_d(points_3d, tri):
    # Define the lifting map and its partial derivatives
    def f(x, y):
        return x**2 + x * y + y**2

    def df_dx(x, y):
        return 2 * x + y

    def df_dy(x, y):
        return x + 2 * y

    # Compute the surface normal
    def surface_normal(x, y):
        fx = df_dx(x, y)
        fy = df_dy(x, y)
        normal = np.array([-fx, -fy, 1])
        normal /= np.linalg.norm(normal)
        return normal

    # Compute normals for all points
    normals = np.array([surface_normal(x, y) for x, y in points_3d[:, :2]])

    # Plot the surface and normals
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, cmap='viridis', alpha=0.8)
    ax.quiver(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], length=0.2, color='r')
    ax.set_title('Lifted Mesh with Surface Normals')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# Part e: Calculate the vertex normal of the lifted mesh and plot it
def part_e(points_3d, tri):
    # Compute face normals
    face_normals = np.zeros((len(tri.simplices), 3))
    for i, simplex in enumerate(tri.simplices):
        v0, v1, v2 = points_3d[simplex]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals[i] = np.cross(edge1, edge2)
        face_normals[i] /= np.linalg.norm(face_normals[i])

    # Compute vertex normals
    vertex_normals = np.zeros_like(points_3d)
    for i, vertex in enumerate(points_3d):
        connected_triangles = np.where(tri.simplices == i)[0]
        vertex_normals[i] = np.mean(face_normals[connected_triangles], axis=0)
        vertex_normals[i] /= np.linalg.norm(vertex_normals[i])

    # Plot the surface and vertex normals
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, cmap='viridis', alpha=0.8)
    ax.quiver(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], vertex_normals[:, 0], vertex_normals[:, 1], vertex_normals[:, 2], length=0.2, color='r')
    ax.set_title('Lifted Mesh with Vertex Normals')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# Part f: Compute the second fundamental form using the vertex normal
def part_f(points_3d, tri):
    # Define the lifting map and its partial derivatives
    def f(x, y):
        return x**2 + x * y + y**2

    def df_dx(x, y):
        return 2 * x + y

    def df_dy(x, y):
        return x + 2 * y

    def d2f_dx2(x, y):
        return 2

    def d2f_dy2(x, y):
        return 2

    def d2f_dxdy(x, y):
        return 1

    # Compute the second fundamental form
    def second_fundamental_form(x, y):
        fx = df_dx(x, y)
        fy = df_dy(x, y)
        denom = np.sqrt(1 + fx**2 + fy**2)
        L = d2f_dx2(x, y) / denom
        M = d2f_dxdy(x, y) / denom
        N = d2f_dy2(x, y) / denom
        return np.array([[L, M], [M, N]])

    # Example: Compute the second fundamental form at a specific point
    x, y = points_3d[0, 0], points_3d[0, 1]
    II = second_fundamental_form(x, y)
    print(f"Second Fundamental Form at (x, y) = ({x}, {y}):")
    print(II)

# Part g: Calculate the shape operator numerically and compute curvatures
def part_g(points_3d, tri):
    # Define the lifting map and its partial derivatives
    def f(x, y):
        return x**2 + x * y + y**2

    def df_dx(x, y):
        return 2 * x + y

    def df_dy(x, y):
        return x + 2 * y

    def d2f_dx2(x, y):
        return 2

    def d2f_dy2(x, y):
        return 2

    def d2f_dxdy(x, y):
        return 1

    # Compute the first fundamental form
    def first_fundamental_form(x, y):
        fx = df_dx(x, y)
        fy = df_dy(x, y)
        return np.array([[1 + fx**2, fx * fy], [fx * fy, 1 + fy**2]])

    # Compute the second fundamental form
    def second_fundamental_form(x, y):
        fx = df_dx(x, y)
        fy = df_dy(x, y)
        denom = np.sqrt(1 + fx**2 + fy**2)
        L = d2f_dx2(x, y) / denom
        M = d2f_dxdy(x, y) / denom
        N = d2f_dy2(x, y) / denom
        return np.array([[L, M], [M, N]])

    # Compute the shape operator
    def shape_operator(x, y):
        I = first_fundamental_form(x, y)
        II = second_fundamental_form(x, y)
        I_inv = np.linalg.inv(I)
        return np.dot(I_inv, II)

    # Compute principal curvatures, Gaussian curvature, and mean curvature
    def compute_curvatures(x, y):
        S = shape_operator(x, y)
        eigenvalues = np.linalg.eigvals(S)
        kappa1, kappa2 = eigenvalues
        K = kappa1 * kappa2  # Gaussian curvature
        H = (kappa1 + kappa2) / 2  # Mean curvature
        return kappa1, kappa2, K, H

    # Compute curvatures for all points
    kappa1_values = np.zeros(len(points_3d))
    kappa2_values = np.zeros(len(points_3d))
    K_values = np.zeros(len(points_3d))
    H_values = np.zeros(len(points_3d))
    for i, (x, y, z) in enumerate(points_3d):
        kappa1, kappa2, K, H = compute_curvatures(x, y)
        kappa1_values[i] = kappa1
        kappa2_values[i] = kappa2
        K_values[i] = K
        H_values[i] = H

    # Plot the surface with curvature visualizations
    fig = plt.figure(figsize=(18, 12))

    # Plot Gaussian curvature
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, facecolors=plt.cm.viridis(K_values), alpha=0.8)
    ax1.set_title('Gaussian Curvature (K)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Plot mean curvature
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, facecolors=plt.cm.viridis(H_values), alpha=0.8)
    ax2.set_title('Mean Curvature (H)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    # Plot principal curvature kappa1
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, facecolors=plt.cm.viridis(kappa1_values), alpha=0.8)
    ax3.set_title('Principal Curvature (κ₁)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')

    # Plot principal curvature kappa2
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, facecolors=plt.cm.viridis(kappa2_values), alpha=0.8)
    ax4.set_title('Principal Curvature (κ₂)')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')

    plt.tight_layout()
    plt.show()

# Main function to run all parts
def main():
    # Part a
    points, tri = part_a()

    # Part b
    points_3d, tri = part_b(points, tri)

    # Part c
    part_c(points)

    # Part d
    part_d(points_3d, tri)

    # Part e
    part_e(points_3d, tri)

    # Part f
    part_f(points_3d, tri)

    # Part g
    part_g(points_3d, tri)

# Run the main function
if __name__ == "__main__":
    main()
