"""
irb120_fk.py
Cinematica directa (FK) por producto de transformaciones del URDF.
Es la misma logica de tu script del Hito 1, reorganizada como modulo.

Entradas : q  -> array (6,) en rad
Salidas  : T  -> matriz homogenea 4x4 (base_link -> gripper)
"""
import numpy as np
from irb120_params import JOINTS, GRIPPER_FIXED_XYZ, GRIPPER_FIXED_RPY


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def homog(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rpy_to_R(rpy):
    """Convencion URDF: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rotz(rpy[2]) @ roty(rpy[1]) @ rotx(rpy[0])


def joint_transform(xyz, rpy, axis, q):
    T_fixed = homog(rpy_to_R(rpy), np.array(xyz, dtype=float))
    R_var = {'x': rotx, 'y': roty, 'z': rotz}[axis](q)
    return T_fixed @ homog(R_var, np.zeros(3))


def forward_kinematics(q):
    T = np.eye(4)
    for i, (xyz, rpy, axis) in enumerate(JOINTS):
        T = T @ joint_transform(xyz, rpy, axis, q[i])
    T = T @ homog(rpy_to_R(GRIPPER_FIXED_RPY), np.array(GRIPPER_FIXED_XYZ, dtype=float))
    return T


def pose_error(T_a, T_b):
    """Devuelve (error de posicion [m], error angular [rad]) entre dos poses."""
    e_pos = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    Rrel = T_a[:3, :3].T @ T_b[:3, :3]
    # atan2 es numericamente estable cerca de 0 (arccos pierde ~1e-8 de precision)
    v = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]]) / 2.0
    return e_pos, float(np.arctan2(np.linalg.norm(v), (np.trace(Rrel) - 1.0) / 2.0))
