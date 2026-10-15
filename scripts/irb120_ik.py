"""
irb120_ik.py
Cinematica inversa analitica (Pieper) para el modelo del URDF del Hito 1.

Estructura que explota Pieper (se verifica en el URDF):
  * Ejes 4, 5, 6 se cortan en un punto: el CENTRO DE LA MUNECA (origen de J5).
    (el offset 0.04 de J6 y 0.072 de la pinza van sobre el eje z6 => herramienta)
  * J1 gira sobre z (vertical); J2 y J3 son paralelos (eje y).
  => posicion (q1,q2,q3) y orientacion (q4,q5,q6) se desacoplan.

Entrada : T_des (4x4) pose deseada de la pinza respecto a base_link
Salida  : lista de dicts {'q': array(6), 'branch': (...), 'singular': bool}
"""
import numpy as np
from irb120_params import JOINTS, GRIPPER_FIXED_XYZ
from irb120_fk import rotz, roty

# ---- Constantes geometricas DERIVADAS de JOINTS (no hardcodeadas) ----------
# Se usan los angulos rpy REALES del URDF (1.5708, no pi/2 exacto), para que
# la IK sea consistente con la FK al nivel de precision de maquina.
P2, P3, P4 = JOINTS[1][1][1], JOINTS[2][1][1], JOINTS[3][1][1]
assert abs(P2 + P3) < 1e-12, "J2 y J3 deben cancelarse para que frame 3 quede alineado"

# Altura del eje de J2 sobre la base
D1 = JOINTS[0][0][2] + JOINTS[1][0][2]

# Vector J2->J3 en el plano del brazo (x,z) del frame 1:  Ry(P2) * [L,0,0]
_L = JOINTS[2][0][0]
UX, UZ = _L * np.cos(P2), -_L * np.sin(P2)
L2 = float(np.hypot(UX, UZ))
BETA0 = float(np.arctan2(UX, UZ))          # ~0: brazo superior vertical en q=0

# Vector J3->centro de muneca en frame 3: offset J4 + Ry(P4) * [0,0,d4]
_d4 = JOINTS[4][0][2]
WX = JOINTS[3][0][0] + _d4 * np.sin(P4)
WZ = _d4 * np.cos(P4)
L3 = float(np.hypot(WX, WZ))
ALPHA0 = float(np.arctan2(WX, WZ))         # angulo del vector J3->muneca en q=0

# Herramienta: desde el centro de muneca hasta la pinza, a lo largo de z6
D_TOOL = JOINTS[5][0][2] + GRIPPER_FIXED_XYZ[2]


def wrap(a):
    """Lleva un angulo a (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi) if abs(a) > np.pi else float(a)


def _wrist_solutions(Rw, eps=1e-9):
    """
    Rw = Rz(q4) Ry(q5) Rz(q6)  (Euler ZYZ).  Devuelve lista de (q4,q5,q6,sing).
    Rw[2,2]=cos q5 ; Rw[0,2]=cos q4 sin q5 ; Rw[1,2]=sin q4 sin q5
    Rw[2,0]=-sin q5 cos q6 ; Rw[2,1]= sin q5 sin q6
    """
    s5 = np.hypot(Rw[0, 2], Rw[1, 2])
    if s5 < eps:                                    # SINGULARIDAD: q5 = 0 o pi
        q5 = 0.0 if Rw[2, 2] > 0 else np.pi
        q4 = 0.0                                    # arbitrario: solo q4(+/-)q6 es observable
        if q5 == 0.0:
            q6 = np.arctan2(Rw[1, 0], Rw[0, 0])     # q4 + q6
        else:
            q6 = -np.arctan2(-Rw[1, 0], -Rw[0, 0])  # q4 - q6
        return [(q4, q5, q6, True)]
    out = []
    # q5 > 0
    out.append((np.arctan2(Rw[1, 2], Rw[0, 2]), np.arctan2(s5, Rw[2, 2]),
                np.arctan2(Rw[2, 1], -Rw[2, 0]), False))
    # q5 < 0 (muneca invertida)
    out.append((np.arctan2(-Rw[1, 2], -Rw[0, 2]), np.arctan2(-s5, Rw[2, 2]),
                np.arctan2(-Rw[2, 1], Rw[2, 0]), False))
    return out


def ik_pieper(T_des, tol=1e-9):
    """
    Devuelve hasta 8 soluciones (2 q1 x 2 codo x 2 muneca).
    Lista vacia => pose fuera del espacio de trabajo.
    """
    R = T_des[:3, :3]
    p = T_des[:3, 3]

    # Paso 1: centro de muneca  p_wc = p - d_tool * z6
    pwc = p - D_TOOL * R[:, 2]
    rho = np.hypot(pwc[0], pwc[1])       # rho ~ 0 => singularidad de hombro (q1 indefinido)
    z = pwc[2] - D1

    sols = []
    for k, sgn in enumerate((+1, -1)):                       # Paso 2: rama de q1
        q1 = np.arctan2(pwc[1], pwc[0]) + (0.0 if sgn > 0 else np.pi)
        r = sgn * rho                                        # radio con signo en el plano del brazo
        # Paso 3: planar 2R (ley de cosenos)
        c = (r * r + z * z - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
        if abs(c) > 1 + tol:
            continue                                         # inalcanzable en esta rama
        c = np.clip(c, -1.0, 1.0)
        for j, s in enumerate((+1, -1)):                     # codo arriba/abajo
            psi = s * np.arccos(c)
            q2p = np.arctan2(r, z) - np.arctan2(L3 * np.sin(psi), L2 + L3 * np.cos(psi))
            q2 = q2p - BETA0
            q3 = psi - ALPHA0 + BETA0
            # Paso 4: orientacion de la muneca
            R03 = rotz(q1) @ roty(q2 + q3)
            Rw = roty(-P4) @ R03.T @ R                 # = Rz(q4)Ry(q5)Rz(q6)
            for m, (q4, q5, q6, sing) in enumerate(_wrist_solutions(Rw)):
                q = np.array([wrap(x) for x in (q1, q2, q3, q4, q5, q6)])
                sols.append({'q': q, 'branch': (k, j, m), 'singular': sing})
    return sols


def filter_by_limits(sols, lo, hi):
    """Descarta soluciones fuera de [lo, hi] (arrays de 6, en rad)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    return [s for s in sols if np.all(s['q'] >= lo) and np.all(s['q'] <= hi)]


def closest_solution(sols, q_ref):
    """Elige la solucion mas cercana (distancia angular envuelta) a q_ref."""
    def dist(s):
        d = (s['q'] - np.asarray(q_ref) + np.pi) % (2 * np.pi) - np.pi
        return np.linalg.norm(d)
    return min(sols, key=dist) if sols else None
