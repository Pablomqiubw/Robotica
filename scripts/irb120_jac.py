"""
irb120_jac.py
Jacobiana geometrica 6x6 del modelo URDF del Hito 1 y utilidades de singularidad.

Convencion: xi = [v ; w] = J(q) * qd
  v : velocidad lineal del punto de referencia (por defecto el origen de la pinza)
  w : velocidad angular del efector
  Ambas expresadas en el frame base_link.

Formula (junta de revolucion i, eje z_i y origen p_i expresados en la base):
    J_v,i = z_i x (p_e - p_i)        J_w,i = z_i

Entradas : q (6,) rad ; point (3,) opcional: punto de referencia en la base
Salidas  : J (6,6)
"""
import numpy as np
from irb120_params import JOINTS, GRIPPER_FIXED_XYZ, GRIPPER_FIXED_RPY
from irb120_fk import rotx, roty, rotz, homog, rpy_to_R

_AXES = {'x': np.array([1.0, 0, 0]), 'y': np.array([0, 1.0, 0]), 'z': np.array([0, 0, 1.0])}
_ROT = {'x': rotx, 'y': roty, 'z': rotz}


def joint_axes_and_origins(q):
    """Devuelve (zs, ps, T_tool): eje y origen de cada junta en la base, y pose de la pinza."""
    T = np.eye(4)
    zs, ps = [], []
    for i, (xyz, rpy, axis) in enumerate(JOINTS):
        T = T @ homog(rpy_to_R(rpy), np.array(xyz, dtype=float))   # parte fija del URDF
        zs.append(T[:3, :3] @ _AXES[axis])
        ps.append(T[:3, 3].copy())
        T = T @ homog(_ROT[axis](q[i]), np.zeros(3))               # giro de la junta
    T = T @ homog(rpy_to_R(GRIPPER_FIXED_RPY), np.array(GRIPPER_FIXED_XYZ, dtype=float))
    return zs, ps, T


def jacobian(q, point=None):
    zs, ps, T = joint_axes_and_origins(q)
    pe = T[:3, 3] if point is None else np.asarray(point, dtype=float)
    J = np.zeros((6, 6))
    for i in range(6):
        J[:3, i] = np.cross(zs[i], pe - ps[i])
        J[3:, i] = zs[i]
    return J


def wrist_center(q):
    """Centro de muneca = origen de J5 (donde se cortan los ejes 4,5,6)."""
    _, ps, _ = joint_axes_and_origins(q)
    return ps[4]


def singular_values(J):
    return np.linalg.svd(J, compute_uv=False)


def wrist_rates_analytic(w_B, q4, q5):
    """
    Velocidades de muneca (qd4, qd5, qd6) a partir de la velocidad angular de
    la muneca w_B expresada en el frame B (justo antes de J4), con Euler ZYZ:
        w_x = -sin q4 qd5 + cos q4 sin q5 qd6
        w_y =  cos q4 qd5 + sin q4 sin q5 qd6
        w_z =  qd4 + cos q5 qd6
    Despejando:
        qd5 = -sin q4 w_x + cos q4 w_y
        qd6 = (cos q4 w_x + sin q4 w_y) / sin q5          <-- diverge si q5 -> 0
        qd4 = w_z - cot(q5) (cos q4 w_x + sin q4 w_y)     <-- diverge si q5 -> 0
    """
    a = np.cos(q4) * w_B[0] + np.sin(q4) * w_B[1]
    qd5 = -np.sin(q4) * w_B[0] + np.cos(q4) * w_B[1]
    qd6 = a / np.sin(q5)
    qd4 = w_B[2] - a * np.cos(q5) / np.sin(q5)
    return np.array([qd4, qd5, qd6])
