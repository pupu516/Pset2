import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# 1. Stereographic projection
# ----------------------------
def stereographic_projection(x, y, z):
    """
    Stereographic projection from the unit sphere (except the north pole z=+1).
    (x, y, z) -> (x', y') = ( x/(1 - z), y/(1 - z) ).
    """
    denom = 1.0 - z
    xprime = x / denom
    yprime = y / denom
    return xprime, yprime

# ----------------------------
# 2. Parametrize three great circles
# ----------------------------
# We'll use t in [0..2π] to trace out each circle on the unit sphere.

# Circle A:  z=0  =>  x=cos(t), y=sin(t), z=0
def circle_A(t):
    return np.cos(t), np.sin(t), 0.0

# Circle B:  x=0  =>  y=cos(t), z=sin(t), x=0
def circle_B(t):
    return 0.0, np.cos(t), np.sin(t)

# Circle C:  y=0  =>  x=cos(t), z=sin(t), y=0
def circle_C(t):
    return np.cos(t), 0.0, np.sin(t)

# We'll sample t from 0..2π for each circle.
t_values = np.linspace(0, 2*np.pi, 300)

# ----------------------------
# 3. Prepare figure & 3D sphere
# ----------------------------
fig = plt.figure(figsize=(12,6))

ax3d = fig.add_subplot(1,2,1, projection='3d')
ax3d.set_title("Great circles on the unit sphere")

ax2d = fig.add_subplot(1,2,2)
ax2d.set_title("Stereographic projection")

# Optional: plot a faint wireframe sphere
u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax3d.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.3)

# ----------------------------
# 4. Plot each circle in 3D
# ----------------------------
def plot_circle_3d(circle_func, t_vals, color, label):
    xlist, ylist, zlist = [], [], []
    for t in t_vals:
        x, y, z = circle_func(t)
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
    ax3d.plot(xlist, ylist, zlist, color=color, label=label)

# Circle A: equator
plot_circle_3d(circle_A, t_values, 'r', "Circle A: z=0")
# Circle B: x=0 plane
plot_circle_3d(circle_B, t_values, 'g', "Circle B: x=0")
# Circle C: y=0 plane
plot_circle_3d(circle_C, t_values, 'b', "Circle C: y=0")

ax3d.set_xlim([-1,1])
ax3d.set_ylim([-1,1])
ax3d.set_zlim([-1,1])
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")
ax3d.view_init(elev=25, azim=45)
ax3d.legend()

# ----------------------------
# 5. Project each circle to 2D
# ----------------------------
def plot_circle_2d(circle_func, t_vals, color, label):
    xlist, ylist = [], []
    for t in t_vals:
        x3, y3, z3 = circle_func(t)
        x2, y2 = stereographic_projection(x3, y3, z3)
        xlist.append(x2)
        ylist.append(y2)
    ax2d.plot(xlist, ylist, color=color, label=label)

plot_circle_2d(circle_A, t_values, 'r', "Circle A: z=0")
plot_circle_2d(circle_B, t_values, 'g', "Circle B: x=0")
plot_circle_2d(circle_C, t_values, 'b', "Circle C: y=0")

ax2d.set_aspect("equal", adjustable="box")
ax2d.set_xlabel("x'")
ax2d.set_ylabel("y'")
ax2d.legend()

plt.tight_layout()
plt.savefig("great_circle_map.png", dpi=1000, bbox_inches='tight')

