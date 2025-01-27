import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------
# 1. Spherical coordinates and Stereographic Projection
# ----------------------------------------------------------
def sphere_xyz(theta, phi):
    """
    Convert spherical coords (theta, phi) -> (x, y, z) on the unit sphere.
    Using the 'physics' convention:
       x = sin(theta)*cos(phi),
       y = sin(theta)*sin(phi),
       z = cos(theta).
    """
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def stereographic_projection(x, y, z):
    """
    (x, y, z) on the sphere -> (x', y') in the plane via
       x' = x / (1 - z),
       y' = y / (1 - z).
    """
    denom = 1.0 - z
    xprime = x / denom
    yprime = y / denom
    return xprime, yprime

# ----------------------------------------------------------
# 2. Define Two Non-Special Curves that Intersect at Two Points
# ----------------------------------------------------------
#
# We'll define each curve in terms of a parameter t in [0,1]:
#   curve_1(t): theta_1(t), phi_1(t)
#   curve_2(t): theta_2(t), phi_2(t)
#
# By choosing the same phi(t) but slightly different theta(t) in a symmetric way,
# we guarantee that at t=0 and t=0.5, they coincide.

def curve_1(t):
    """
    Curve 1 on the sphere:
      theta_1(t) = 0.5 + 0.1*sin(2πt)
      phi_1(t)   = 2πt
    """
    theta = 0.5 + 0.1 * np.sin(2*np.pi*t)
    phi   = 2*np.pi * t
    return sphere_xyz(theta, phi)

def curve_2(t):
    """
    Curve 2 on the sphere:
      theta_2(t) = 0.5 - 0.1*sin(2πt)
      phi_2(t)   = 2πt
    """
    theta = 0.5 - 0.1 * np.sin(2*np.pi*t)
    phi   = 2*np.pi * t
    return sphere_xyz(theta, phi)

# These two curves intersect whenever sin(2πt) = 0 => t = 0, 0.5, 1.0, etc.
# We'll use t=0 and t=0.5 as two distinct intersection points on the sphere.

# ----------------------------------------------------------
# 3. Numerical Derivatives for Angle Checking
# ----------------------------------------------------------
def derivative_curve(curve_func, t0, eps=1e-6):
    """
    Numerically approximate the 3D tangent vector d/dt of curve_func(t) at t = t0.
    curve_func(t) returns (x,y,z) on the sphere.
    """
    x0, y0, z0 = curve_func(t0)
    x1, y1, z1 = curve_func(t0 + eps)
    return np.array([x1 - x0, y1 - y0, z1 - z0]) / eps

# ----------------------------------------------------------
# 4. Plotting Setup
# ----------------------------------------------------------
fig = plt.figure(figsize=(12, 6))

# Left subplot: 3D sphere
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.set_title("Two curves intersecting on the sphere")

# Right subplot: Stereographic plane
ax2d = fig.add_subplot(1, 2, 2)
ax2d.set_title("Stereographic projection")

# Optional: wireframe for the sphere in 3D
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax3d.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.3)

# ----------------------------------------------------------
# 5. Plot the Curves in 3D and 2D
# ----------------------------------------------------------
t_vals = np.linspace(0, 1, 300)

# Curve 1
x1_list, y1_list, z1_list = [], [], []
for t in t_vals:
    x, y, z = curve_1(t)
    x1_list.append(x); y1_list.append(y); z1_list.append(z)

ax3d.plot(x1_list, y1_list, z1_list, 'r', label='Curve 1')

x1p, y1p = stereographic_projection(np.array(x1_list),
                                    np.array(y1_list),
                                    np.array(z1_list))
ax2d.plot(x1p, y1p, 'r', label='Curve 1')

# Curve 2
x2_list, y2_list, z2_list = [], [], []
for t in t_vals:
    x, y, z = curve_2(t)
    x2_list.append(x); y2_list.append(y); z2_list.append(z)

ax3d.plot(x2_list, y2_list, z2_list, 'b', label='Curve 2')

x2p, y2p = stereographic_projection(np.array(x2_list),
                                    np.array(y2_list),
                                    np.array(z2_list))
ax2d.plot(x2p, y2p, 'b', label='Curve 2')

# ----------------------------------------------------------
# 6. Check Angles at Multiple Intersection Points
# ----------------------------------------------------------
# The curves intersect when sin(2πt)=0 => t=0, 0.5, 1.0, ...
# We'll check t=0 and t=0.5 as two distinct intersections.
intersection_params = [0.0, 0.5]

for i, t_int in enumerate(intersection_params, start=1):

    # 3D intersection coords from curve_1(t_int) == curve_2(t_int)
    xI1, yI1, zI1 = curve_1(t_int)  # same as curve_2(t_int)
    
    # 3D tangents
    dC1_3d = derivative_curve(curve_1, t_int)
    dC2_3d = derivative_curve(curve_2, t_int)
    dot_3d = np.dot(dC1_3d, dC2_3d)
    angle_3d = np.arccos(
        dot_3d / (np.linalg.norm(dC1_3d)*np.linalg.norm(dC2_3d))
    )
    angle_3d_deg = np.degrees(angle_3d)
    
    # 2D intersection coords
    xIp, yIp = stereographic_projection(xI1, yI1, zI1)

    # 2D tangents (finite difference around t_int)
    eps = 1e-6
    # curve_1 step
    x1_eps, y1_eps, z1_eps = curve_1(t_int + eps)
    x1p_eps, y1p_eps = stereographic_projection(x1_eps, y1_eps, z1_eps)
    dC1_2d = np.array([x1p_eps - xIp, y1p_eps - yIp]) / eps
    
    # curve_2 step
    x2_eps, y2_eps, z2_eps = curve_2(t_int + eps)
    x2p_eps, y2p_eps = stereographic_projection(x2_eps, y2_eps, z2_eps)
    dC2_2d = np.array([x2p_eps - xIp, y2p_eps - yIp]) / eps

    dot_2d = np.dot(dC1_2d, dC2_2d)
    angle_2d = np.arccos(
        dot_2d / (np.linalg.norm(dC1_2d)*np.linalg.norm(dC2_2d))
    )
    angle_2d_deg = np.degrees(angle_2d)
    
    print(f"\nIntersection #{i} (t={t_int}):")
    print(f"  Angle in 3D = {angle_3d_deg:.2f} degrees")
    print(f"  Angle in 2D = {angle_2d_deg:.2f} degrees")

    # Plot intersection on the sphere
    ax3d.scatter([xI1], [yI1], [zI1], c='k', s=50)
    ax3d.text(xI1, yI1, zI1,
              f"{angle_3d_deg:.1f}°",
              color='k', fontsize=9)

    # Plot intersection in 2D
    ax2d.scatter([xIp], [yIp], c='k', s=50)
    ax2d.text(xIp, yIp,
              f"{angle_2d_deg:.1f}°",
              color='k', fontsize=9)

# ----------------------------------------------------------
# 7. Final Plot Adjustments
# ----------------------------------------------------------
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.set_xlim([-1,1])
ax3d.set_ylim([-1,1])
ax3d.set_zlim([-1,1])
ax3d.view_init(elev=25, azim=45)

ax2d.set_aspect("equal", adjustable="box")
ax2d.set_xlabel("x'")
ax2d.set_ylabel("y'")

# Make a legend on the 2D plot (avoid repeated labels)
handles, labels = ax2d.get_legend_handles_labels()
unique_handles, unique_labels = [], []
for h, l in zip(handles, labels):
    if l not in unique_labels:
        unique_handles.append(h)
        unique_labels.append(l)
ax2d.legend(unique_handles, unique_labels)

plt.tight_layout()
plt.savefig("angle_check.png", dpi=1000, bbox_inches='tight')

