import numpy as np
import matplotlib.pyplot as plt

theta0 = np.pi / 5  
theta_final = np.pi / 2  
t_points = 20


alpha = 1  
beta = 1  
magnitude = np.sqrt(alpha**2 + beta**2 * np.sin(theta0)**2)  


theta = np.linspace(theta0, theta_final, t_points)
phi = np.zeros(t_points)  


n_theta = np.zeros(t_points)
n_phi = np.zeros(t_points)


n_theta[0] = alpha
n_phi[0] = beta

for i in range(1, t_points):
    n_theta[i] = n_theta[i - 1]
    n_phi[i] = n_phi[i - 1]

    norm = np.sqrt(n_theta[i]**2 + n_phi[i]**2 * np.sin(theta[i])**2)
    n_theta[i] /= norm
    n_phi[i] /= norm

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

nx = n_theta * np.cos(theta) - n_phi * np.sin(phi)
ny = n_theta * np.sin(phi) + n_phi * np.cos(phi)
nz = -n_theta * np.sin(theta)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    np.outer(np.sin(np.linspace(0, np.pi, 50)), np.cos(np.linspace(0, 2 * np.pi, 50))),
    np.outer(np.sin(np.linspace(0, np.pi, 50)), np.sin(np.linspace(0, 2 * np.pi, 50))),
    np.outer(np.cos(np.linspace(0, np.pi, 50)), np.ones(50)),
    color='lightblue',
    alpha=0.5
)
ax.quiver(x, y, z, nx, ny, nz, color='red', length=0.1, normalize=True)
ax.set_title("Parallel Transport of a Vector on a Sphere")
plt.savefig("parallel_transport.png", dpi=300, bbox_inches='tight')




