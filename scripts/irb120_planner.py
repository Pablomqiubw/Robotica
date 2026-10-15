"""
irb120_planner.py
Planificador SIN ROS:  pose deseada -> IK de Pieper -> eleccion de rama -> LSPB.
El nodo ROS solo es una capa fina sobre este modulo (asi se prueba sin ROS).

Entrada principal : plan_to_pose(T_des, q_start, vmax, amax, lo, hi)
Salida            : dict con ok / reason / q_goal / branch / trayectoria / errores FK

IMPORTANTE (simulacion vs robot real):
  * Todo lo que calcula este modulo es MATEMATICO (simulacion). Que la IK reproduzca la
    pose por FK demuestra consistencia con el URDF, NO precision del robot fisico.
  * La trayectoria LSPB es recta en el ESPACIO ARTICULAR: la pinza NO sigue una recta
    cartesiana entre inicio y meta.
  * Los limites (lo, hi, vmax, amax) los entrega el usuario / el URDF; aqui no se inventan.
"""
import xml.etree.ElementTree as ET
import numpy as np
from irb120_fk import forward_kinematics, pose_error
from irb120_ik import ik_pieper
from irb120_jac import jacobian, singular_values
from irb120_traj import LSPB

TWO_PI = 2.0 * np.pi


# ----------------------------- pose <-> matriz -------------------------------
def quat_to_R(x, y, z, w):
    """Cuaternion (x,y,z,w) -> matriz de rotacion 3x3. Se normaliza; norma ~0 es error."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        raise ValueError("cuaternion nulo")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """Matriz de rotacion -> cuaternion (x,y,z,w) con w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q if q[3] >= 0 else -q


def pose_to_T(position, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*quat_xyzw)
    T[:3, 3] = position
    return T


# ----------------------------- lectura del URDF -------------------------------
def parse_urdf_joints(urdf_xml):
    """
    Lee del URDF (texto XML ya procesado, sin xacro) las juntas MOVILES en el orden de
    la cadena base -> pinza.  Devuelve dict con:
      names (list), lower, upper (arrays; -inf/inf si la junta no declara limite),
      velocity (array; nan si no declara), has_pos_limits, has_vel_limits (bool)
    Nada se inventa: lo que el URDF no declara queda como inf / nan.
    """
    root = ET.fromstring(urdf_xml)
    joints = []
    for j in root.findall('joint'):
        lim = j.find('limit')
        get = (lambda k: float(lim.get(k)) if lim is not None and lim.get(k) is not None else None)
        joints.append({'name': j.get('name'), 'type': j.get('type'),
                       'parent': j.find('parent').get('link'), 'child': j.find('child').get('link'),
                       'lower': get('lower'), 'upper': get('upper'), 'velocity': get('velocity')})
    children = {j['child'] for j in joints}
    roots = [p for p in {j['parent'] for j in joints} if p not in children]
    if len(roots) != 1:
        raise ValueError(f"se esperaba una cadena con una sola raiz, hay {roots}")
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j['parent'], []).append(j)
    ordered, stack = [], [roots[0]]
    while stack:
        link = stack.pop()
        for j in by_parent.get(link, []):
            ordered.append(j)
            stack.append(j['child'])
    mov = [j for j in ordered if j['type'] in ('revolute', 'continuous', 'prismatic')]
    lower = np.array([j['lower'] if j['lower'] is not None else -np.inf for j in mov])
    upper = np.array([j['upper'] if j['upper'] is not None else np.inf for j in mov])
    vel = np.array([j['velocity'] if j['velocity'] is not None else np.nan for j in mov])
    return {'names': [j['name'] for j in mov], 'lower': lower, 'upper': upper, 'velocity': vel,
            'has_pos_limits': bool(np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))),
            'has_vel_limits': bool(np.all(np.isfinite(vel)))}


# ----------------------------- planificador -----------------------------------
def _nearest_representative(q_sol, q_ref, lo, hi):
    """
    Cada angulo vale igual +-2*pi. Elige, por junta, el representante DENTRO de [lo, hi]
    mas cercano a q_ref (asi el LSPB no da vueltas innecesarias). Devuelve None si alguna
    junta no tiene ningun representante dentro de limites.
    """
    out = np.empty(6)
    for i in range(6):
        k0 = np.round((q_ref[i] - q_sol[i]) / TWO_PI)
        cands = [q_sol[i] + TWO_PI * k for k in (k0 - 1, k0, k0 + 1)]
        cands = [c for c in cands if lo[i] - 1e-9 <= c <= hi[i] + 1e-9]
        if not cands:
            return None
        out[i] = min(cands, key=lambda c: abs(c - q_ref[i]))
    return out


def _fail(reason, **extra):
    d = {'ok': False, 'reason': reason, 'warnings': []}
    d.update(extra)
    return d


def plan_to_pose(T_des, q_start, vmax, amax, lo=None, hi=None, dt=0.01,
                 fk_tol=1e-6, sing_tol=5e-3):
    """
    T_des   : (4,4) pose deseada de la pinza en base_link
    q_start : (6,) configuracion actual [rad]
    vmax, amax : (6,) limites de velocidad [rad/s] y aceleracion [rad/s^2]
    lo, hi  : (6,) limites articulares [rad] (None o +-inf = sin limite)
    Devuelve dict: ok, reason, warnings, q_goal, branch, n_solutions, n_valid, lspb, t, Q, Qd, Qdd,
                   fk_err_pos, fk_err_ang, min_sigma_path, near_singularity
    """
    q_start = np.asarray(q_start, dtype=float)
    lo = np.full(6, -np.inf) if lo is None else np.asarray(lo, dtype=float)
    hi = np.full(6, np.inf) if hi is None else np.asarray(hi, dtype=float)
    warnings = []
    if np.any(q_start < lo - 1e-9) or np.any(q_start > hi + 1e-9):
        warnings.append("q_start esta fuera de los limites articulares")

    sols = ik_pieper(T_des)
    if not sols:
        return _fail("pose fuera del espacio de trabajo (Pieper no devuelve soluciones)", n_solutions=0)

    cands = []
    for s in sols:
        q = _nearest_representative(s['q'], q_start, lo, hi)
        if q is None:
            continue
        e_p, e_a = pose_error(forward_kinematics(q), T_des)     # validacion FK de CADA candidata
        if e_p < fk_tol and e_a < fk_tol:
            cands.append({'q': q, 'branch': s['branch'], 'singular': s['singular'], 'e_p': e_p, 'e_a': e_a})
    if not cands:
        return _fail("ninguna solucion cumple limites articulares y validacion FK", n_solutions=len(sols), n_valid=0)

    best = min(cands, key=lambda c: np.linalg.norm(c['q'] - q_start))
    lspb = LSPB(q_start, best['q'], vmax, amax)
    t, Q, Qd, Qdd = lspb.sample(dt)

    sig = np.array([singular_values(jacobian(q))[-1] for q in Q[:: max(1, len(Q) // 200)]])
    min_sig = float(sig.min())
    near = bool(best['singular'] or min_sig < sing_tol)
    if near:
        warnings.append(f"cerca de singularidad de muneca (sigma_min = {min_sig:.2e}); "
                        f"velocidades/torques articulares pueden crecer ~1/sin(q5)")
    return {'ok': True, 'reason': '', 'warnings': warnings, 'q_goal': best['q'], 'branch': best['branch'],
            'n_solutions': len(sols), 'n_valid': len(cands), 'lspb': lspb, 't': t, 'Q': Q, 'Qd': Qd, 'Qdd': Qdd,
            'fk_err_pos': best['e_p'], 'fk_err_ang': best['e_a'],
            'min_sigma_path': min_sig, 'near_singularity': near}
