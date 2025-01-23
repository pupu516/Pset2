import numpy as np

def local_coor_gen(x: np.ndarray, y: np.ndarray, z: np.ndarray):

    dz_dx = np.gradient(z, axis=1) / np.gradient(x, axis=1)
    dz_dy = np.gradient(z, axis=0) / np.gradient(y, axis=0)

    tangent_x = np.stack((np.ones_like(z), np.zeros_like(z), dz_dx), axis=-1)
    tangent_y = np.stack((np.zeros_like(z), np.ones_like(z), dz_dy), axis=-1)

    t_x = tangent_x / np.linalg.norm(t_x, axis=-1, keepdims=True)
    t_y = tangent_y / np.linalg.norm(t_y, axis=-1, keepdims=True)

    normal = np.cross(tangent_x, tangent_y, axis=-1)
    n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

    return t_x, t_y, n



