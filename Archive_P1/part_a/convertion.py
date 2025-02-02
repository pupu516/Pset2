import numpy as np

# Spherical to Cartesian coordinates
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Cartesian to Spherical coordinates
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

# Cylindrical to Cartesian coordinates
def cylindrical_to_cartesian(rho, psi, z):
    x = rho * np.cos(psi)
    y = rho * np.sin(psi)
    return x, y, z

# Cartesian to Cylindrical coordinates
def cartesian_to_cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    psi = np.arctan2(y, x)
    return rho, psi, z

# Spherical basis vectors in terms of Cartesian basis
def spherical_basis_vectors(theta, phi):
    e_r = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    e_theta = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    e_phi = np.array([
        -np.sin(phi),
        np.cos(phi),
        0
    ])
    return e_r, e_theta, e_phi

# Cylindrical basis vectors in terms of Cartesian basis
def cylindrical_basis_vectors(psi):
    e_rho = np.array([np.cos(psi), np.sin(psi), 0])
    e_psi = np.array([-np.sin(psi), np.cos(psi), 0])
    e_z = np.array([0, 0, 1])
    return e_rho, e_psi, e_z
