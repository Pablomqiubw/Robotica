#!/usr/bin/env python3
"""
hito2_seguridad.py - Hito 2 completo en tiempo real para el ABB IRB 120-3/0.6 real:
  1) Cinematica Inversa analitica (Pieper, 8 ramas)
  2) Trayectoria LSPB sincronizada con control de velocidad (v_tcp, v_ori, a_max)
  3) Jacobiano Geometrico 6x6, analisis de singularidades y velocidades cartesianas
  4) Dinamica Inversa por Newton-Euler Recursivo (RNE): tau = M(q)qdd + C(q,qd)qd + g(q)
  5) Validacion combinada de seguridad Hito 1 + Hito 2 para RViz y CLI.

Reutiliza los parametros reales verificados de hito1_seguridad.py:
  d1 = 0.290 m, L2 = 0.270 m, a3 = 0.070 m, d4 = 0.302 m, d6 = 0.072 m
"""
import math
import numpy as np

from hito1_seguridad import (
    D1, L2, A3, D4, D6,
    JOINT_LIMITS_DEG,
    rx, ry, rz, homog,
    fk_detallada_deg,
    evaluar_seguridad as evaluar_seguridad_hito1,
    formatear_reporte_hito1,
)

# Limites oficiales de velocidad articular ABB IRB 120-3/0.6 [deg/s]
JOINT_VMAX_ABS_DEG = np.array([250.0, 250.0, 250.0, 320.0, 320.0, 420.0], dtype=float)
# Pares motores de referencia conservadores para supervision [N*m]
JOINT_TAU_REF_NM = np.array([60.0, 60.0, 40.0, 16.0, 16.0, 9.0], dtype=float)

SPEED_TCP_RANGE = (5.0, 150.0)     # mm/s en Manual Reducido (< 250 mm/s)
SPEED_ORI_RANGE = (5.0, 60.0)      # deg/s orientacion segura
AMAX_RANGE_DEG = (5.0, 180.0)      # deg/s^2 aceleracion LSPB

# Cadena cinematica real IRB 120-3/0.6 (marco i respecto a i-1: offset xyz + giro eje)
REAL_CHAIN = [
    (np.array([0.0, 0.0, 0.0]), 'z'),
    (np.array([0.0, 0.0, D1]), 'y'),
    (np.array([0.0, 0.0, L2]), 'y'),
    (np.array([0.0, 0.0, A3]), 'x'),
    (np.array([D4, 0.0, 0.0]), 'y'),
    (np.array([D6, 0.0, 0.0]), 'x'),
]

_AXES = {
    'x': np.array([1.0, 0.0, 0.0]),
    'y': np.array([0.0, 1.0, 0.0]),
    'z': np.array([0.0, 0.0, 1.0]),
}
_ROT_FN = {'x': rx, 'y': ry, 'z': rz}
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)


def _wrap(a):
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def fk_real_rad(q_rad):
    """Cinematica directa rapida (4x4) en radianes para el modelo ABB IRB 120-3/0.6."""
    T = np.eye(4, dtype=float)
    for (xyz, axis), qi in zip(REAL_CHAIN, q_rad):
        T = T @ homog(_ROT_FN[axis](float(qi)), xyz)
    return T


def _rot_err_deg(R1, R2):
    dR = R1.T @ R2
    v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return math.degrees(math.atan2(0.5 * float(np.linalg.norm(v)), 0.5 * float(np.trace(dR) - 1.0)))


# =============================================================================
# 1. CINEMATICA INVERSA ANALITICA (PIEPER) EN EL MODELO REAL ABB IRB 120-3/0.6
# =============================================================================
def ik_pieper_todas(T_des, q_ref_rad=None, tol=1e-6):
    """
    Resuelve analiticamente (Pieper) todas las ramas validas dentro de JOINT_LIMITS_DEG.
    Retorna dict con 'sols' (lista ordenada por cercania a q_ref_rad),
    'chosen' (mejor solucion o None), 'p_wc_m' y 'alcanzable' (bool).
    """
    R = np.asarray(T_des[:3, :3], dtype=float)
    p = np.asarray(T_des[:3, 3], dtype=float)

    # Desacople posicion / orientacion: centro de muneca (interseccion ejes 4, 5, 6)
    pwc = p - D6 * R[:, 0]
    L3_eff = math.hypot(D4, A3)
    alpha0 = math.atan2(D4, A3)

    r_xy = math.hypot(pwc[0], pwc[1])
    if q_ref_rad is None:
        q_ref = np.zeros(6, dtype=float)
    else:
        q_ref = np.asarray(q_ref_rad, dtype=float)

    if r_xy < 1e-6:
        q1_cands = [('Frente (sing. hombro)', float(q_ref[0]))]
    else:
        q1_f = math.atan2(pwc[1], pwc[0])
        q1_b = _wrap(q1_f + math.pi)
        q1_cands = [('Hombro frente', q1_f), ('Hombro atras', q1_b)]

    limits_rad = [(math.radians(lo), math.radians(hi)) for lo, hi in JOINT_LIMITS_DEG]
    sols = []

    for hombro_lbl, q1 in q1_cands:
        r = math.cos(q1) * pwc[0] + math.sin(q1) * pwc[1]
        z = pwc[2] - D1
        cphi = (r * r + z * z - L2 * L2 - L3_eff * L3_eff) / (2.0 * L2 * L3_eff)
        if abs(cphi) > 1.0 + tol:
            continue
        cphi = max(-1.0, min(1.0, cphi))
        sphi = math.sqrt(max(0.0, 1.0 - cphi * cphi))

        for codo_lbl, sgn_elbow in (('Codo arriba', 1.0), ('Codo abajo', -1.0)):
            phi = math.atan2(sgn_elbow * sphi, cphi)
            q3 = _wrap(phi - alpha0)
            k1 = L2 + L3_eff * math.cos(phi)
            k2 = L3_eff * math.sin(phi)
            q2 = _wrap(math.atan2(r, z) - math.atan2(k2, k1))

            R3 = rz(q1) @ ry(q2 + q3)
            Rw = R3.T @ R
            c5 = max(-1.0, min(1.0, float(Rw[0, 0])))
            s5_mag = math.hypot(float(Rw[1, 0]), float(Rw[2, 0]))

            if s5_mag < 1e-6:
                q5 = 0.0 if c5 > 0 else math.pi
                q4 = float(q_ref[3])
                if q5 == 0.0:
                    q6 = _wrap(math.atan2(float(Rw[2, 1]), float(Rw[1, 1])) - q4)
                else:
                    q6 = _wrap(math.atan2(-float(Rw[2, 1]), -float(Rw[1, 1])) + q4)
                wrist_cands = [('Muneca sing.', q4, q5, q6, True)]
            else:
                wrist_cands = []
                for mun_lbl, s5_sgn in (('Muneca +', 1.0), ('Muneca -', -1.0)):
                    s5 = s5_sgn * s5_mag
                    q5 = math.atan2(s5, c5)
                    q4 = math.atan2(float(Rw[1, 0]) / s5, -float(Rw[2, 0]) / s5)
                    q6 = math.atan2(float(Rw[0, 1]) / s5, float(Rw[0, 2]) / s5)
                    wrist_cands.append((mun_lbl, q4, q5, q6, abs(s5) < 0.05))

            for mun_lbl, q4, q5, q6, is_sing in wrist_cands:
                q6_near = q6 + round((q_ref[5] - q6) / (2.0 * math.pi)) * (2.0 * math.pi)
                seen_q6 = []
                for q6_cand in (q6, q6_near):
                    if any(abs(q6_cand - prev) < 1e-6 for prev in seen_q6):
                        continue
                    seen_q6.append(q6_cand)
                    q_rad = np.array([q1, q2, q3, q4, q5, q6_cand], dtype=float)
                    in_lim = all(lo <= v <= hi for v, (lo, hi) in zip(q_rad, limits_rad))
                    if not in_lim:
                        continue
                    T_chk = fk_real_rad(q_rad)
                    err_mm = float(np.linalg.norm(T_chk[:3, 3] - p) * 1000.0)
                    err_deg = _rot_err_deg(T_chk[:3, :3], R)
                    dist = float(np.linalg.norm(q_rad - q_ref))
                    sols.append({
                        'q_rad': q_rad,
                        'q_deg': [math.degrees(v) for v in q_rad],
                        'branch': (hombro_lbl, codo_lbl, mun_lbl),
                        'singular': bool(is_sing),
                        'err_pos_mm': err_mm,
                        'err_ori_deg': err_deg,
                        'dist_rad': dist,
                    })

    sols.sort(key=lambda s: s['dist_rad'])
    return {
        'sols': sols,
        'chosen': sols[0] if sols else None,
        'p_wc_mm': pwc * 1000.0,
        'L3_eff_mm': L3_eff * 1000.0,
        'alpha0_deg': math.degrees(alpha0),
        'alcanzable': len(sols) > 0,
    }


# =============================================================================
# 2. TRAYECTORIA LSPB SINCRONIZADA CON VELOCIDAD AJUSTABLE
# =============================================================================
def velocidades_lspb_desde_usuario(v_tcp_mms, v_ori_degs, amax_degs2):
    """
    Convierte la velocidad cartesiana del TCP [mm/s], reorientacion [deg/s] y
    aceleracion articular [deg/s^2] a limites articulares seguros (vmax, amax).
    Brazo J1..J3: R_ref = 0.45 m -> v_arm_deg = v_tcp_mm / (450 * pi/180).
    Muneca J4..J6: v_ori_deg.
    """
    v_tcp = max(SPEED_TCP_RANGE[0], min(SPEED_TCP_RANGE[1], float(v_tcp_mms)))
    v_ori = max(SPEED_ORI_RANGE[0], min(SPEED_ORI_RANGE[1], float(v_ori_degs)))
    a_deg = max(AMAX_RANGE_DEG[0], min(AMAX_RANGE_DEG[1], float(amax_degs2)))

    r_ref_mm_per_deg = 450.0 * math.pi / 180.0   # ~7.854 mm por grado en brazo nominal
    v_arm_deg = min(250.0, max(2.0, v_tcp / r_ref_mm_per_deg))
    vmax_deg = np.array([
        min(JOINT_VMAX_ABS_DEG[0], v_arm_deg),
        min(JOINT_VMAX_ABS_DEG[1], v_arm_deg),
        min(JOINT_VMAX_ABS_DEG[2], v_arm_deg),
        min(JOINT_VMAX_ABS_DEG[3], v_ori),
        min(JOINT_VMAX_ABS_DEG[4], v_ori),
        min(JOINT_VMAX_ABS_DEG[5], v_ori * 1.25),
    ], dtype=float)
    amax_deg = np.full(6, a_deg, dtype=float)
    return {
        'v_tcp_mms': v_tcp,
        'v_ori_degs': v_ori,
        'amax_degs2': a_deg,
        'vmax_deg': vmax_deg,
        'amax_deg': amax_deg,
        'vmax_rad': np.radians(vmax_deg),
        'amax_rad': np.radians(amax_deg),
    }


class LSPBReal:
    """Trayectoria LSPB sincronizada multi-eje para el robot real."""

    def __init__(self, q0_rad, qf_rad, vmax_rad, amax_rad):
        self.q0 = np.asarray(q0_rad, dtype=float)
        self.qf = np.asarray(qf_rad, dtype=float)
        self.dq = self.qf - self.q0
        vmax = np.asarray(vmax_rad, dtype=float)
        amax = np.asarray(amax_rad, dtype=float)
        if np.any(vmax <= 0.0) or np.any(amax <= 0.0):
            raise ValueError('vmax y amax deben ser mayores a cero')

        ad = np.abs(self.dq)
        self.vr = float(np.max(ad / vmax))
        self.ar = float(np.max(ad / amax))
        self.junta_critica_v = int(np.argmax(ad / vmax)) + 1
        self.junta_critica_a = int(np.argmax(ad / amax)) + 1

        if self.vr < 1e-12:
            self.D = 0.0
            self.tb = 0.0
            self.T = 0.0
            self.V = 0.0
            self.A = 0.0
            self.tipo = 'sin movimiento'
            return

        sq_ar = math.sqrt(self.ar)
        self.D = max(self.vr, sq_ar)
        self.tb = self.ar / self.D
        self.T = self.D + self.tb
        self.A = 1.0 / (self.tb * self.D)
        self.V = 1.0 / self.D
        self.tipo = 'trapezoidal' if self.vr >= sq_ar - 1e-12 else 'triangular'

    def scalar_at(self, t):
        if self.T <= 0.0:
            return 1.0, 0.0, 0.0
        t = max(0.0, min(self.T, float(t)))
        if t < self.tb:
            return 0.5 * self.A * t * t, self.A * t, self.A
        if t <= self.T - self.tb:
            return 0.5 * self.A * self.tb * self.tb + self.V * (t - self.tb), self.V, 0.0
        tau = self.T - t
        return 1.0 - 0.5 * self.A * tau * tau, self.A * tau, -self.A

    def at(self, t):
        if self.T <= 0.0:
            return self.q0.copy(), np.zeros(6), np.zeros(6)
        s, sd, sdd = self.scalar_at(t)
        return self.q0 + s * self.dq, sd * self.dq, sdd * self.dq

    def sample(self, dt=0.02, min_steps=25):
        if self.T <= 0.0:
            t = np.array([0.0])
            Q = self.q0.reshape(1, 6).copy()
            Z = np.zeros((1, 6))
            return t, Q, Z, Z
        n = max(int(math.ceil(self.T / dt)), min_steps)
        t = np.linspace(0.0, self.T, n + 1)
        out = [self.at(ti) for ti in t]
        Q = np.array([o[0] for o in out])
        Qd = np.array([o[1] for o in out])
        Qdd = np.array([o[2] for o in out])
        return t, Q, Qd, Qdd

    def sample_keyframes(self, n_steps=21):
        """Muestreo rapido pero exacto en transiciones (0, tb, T/2, T-tb, T) para GUI en vivo."""
        if self.T <= 0.0:
            t = np.array([0.0])
            Q = self.q0.reshape(1, 6).copy()
            Z = np.zeros((1, 6))
            return t, Q, Z, Z
        eps = min(1e-4, 0.05 * self.tb) if self.tb > 0.0 else 0.0
        extra = [
            0.0,
            0.5 * self.tb,
            max(0.0, self.tb - eps),
            min(self.T, self.tb + eps),
            0.5 * self.T,
            max(0.0, self.T - self.tb - eps),
            min(self.T, self.T - self.tb + eps),
            self.T - 0.5 * self.tb,
            self.T,
        ]
        grid = np.unique(np.concatenate([np.linspace(0.0, self.T, n_steps), np.array(extra)]))
        out = [self.at(float(ti)) for ti in grid]
        Q = np.array([o[0] for o in out])
        Qd = np.array([o[1] for o in out])
        Qdd = np.array([o[2] for o in out])
        return grid, Q, Qd, Qdd


def _cross3(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], dtype=float)


# =============================================================================
# 3. JACOBIANO GEOMETRICO 6x6 Y SINGULARIDADES EN EL MODELO REAL
# =============================================================================
def joint_axes_origins_real(q_rad):
    """Retorna (zs[6], ps[6], T0^6) en base_link para el ABB IRB 120-3/0.6."""
    zs, ps = [], []
    T = np.eye(4, dtype=float)
    for (xyz, axis), qi in zip(REAL_CHAIN, q_rad):
        T = T @ homog(np.eye(3), xyz)
        zs.append(T[:3, :3] @ _AXES[axis])
        ps.append(T[:3, 3].copy())
        T = T @ homog(_ROT_FN[axis](float(qi)), np.zeros(3))
    return zs, ps, T


def jacobiano_geometrico_rad(q_rad, point=None):
    """
    Jacobiano geometrico 6x6 en base_link: xi = [v; w] = J(q) * qd
      J_v[:, i] = z_i x (p_e - p_i)   [m / rad]
      J_w[:, i] = z_i                  [rad / rad]
    """
    zs, ps, T = joint_axes_origins_real(q_rad)
    pe = T[:3, 3] if point is None else np.asarray(point, dtype=float)
    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        J[:3, i] = _cross3(zs[i], pe - ps[i])
        J[3:, i] = zs[i]
    return J


def analizar_jacobiano_rad(q_rad):
    """Calcula J(q), SVD, determinante, condicionamiento y singularidades."""
    J = jacobiano_geometrico_rad(q_rad)
    s = np.linalg.svd(J, compute_uv=False)
    s_max = float(s[0])
    s_min = float(s[-1])
    det_j = float(np.linalg.det(J))
    cond_j = s_max / s_min if s_min > 1e-12 else float('inf')

    # Indicadores geometricos de singularidad del manipulador 6R
    _, ps, _ = joint_axes_origins_real(q_rad)
    pwc = ps[4]
    rho_hombro_mm = float(math.hypot(pwc[0], pwc[1]) * 1000.0)
    alpha0 = math.atan2(D4, A3)
    sin_codo = abs(math.sin(float(q_rad[2]) + alpha0))
    sin_muneca = abs(math.sin(float(q_rad[4])))

    alertas = []
    if sin_muneca < 0.05:
        alertas.append(
            f'Singularidad de muneca (q5={math.degrees(q_rad[4]):.2f}°, |sin(q5)|={sin_muneca:.4f}; ejes J4 y J6 casi colineales)'
        )
    if sin_codo < 0.05:
        alertas.append(f'Cerca de singularidad de codo (|sin(q3+a0)|={sin_codo:.4f}, brazo extendido/plegado)')
    if rho_hombro_mm < 15.0:
        alertas.append(f'Cerca de singularidad de hombro (rho_muneca={rho_hombro_mm:.1f} mm < 15 mm)')

    return {
        'J': J,
        'singular_values': s,
        'sigma_max': s_max,
        'sigma_min': s_min,
        'det': det_j,
        'cond': cond_j,
        'manipulabilidad': abs(det_j),
        'sin_muneca': sin_muneca,
        'sin_codo': sin_codo,
        'rho_hombro_mm': rho_hombro_mm,
        'en_singularidad': len(alertas) > 0,
        'alertas': alertas,
    }


# =============================================================================
# 4. DINAMICA INVERSA RECURSIVA DE NEWTON-EULER (RNE) EN EL MODELO REAL
# =============================================================================
def irb120_ref_dyn_params(payload_kg=0.0):
    """
    Parametros inerciales nominales de referencia para el ABB IRB 120-3/0.6
    (~25 kg peso total datasheet: base ~10.8 kg fija + ~14.2 kg en links 1..6 + carga util).
    Tensor de inercia respecto al CoM en el marco del link [kg m^2].
    """
    specs = [
        # (masa_kg, com_en_frame_i_m, longitud_ef_m)
        (3.5, np.array([0.0, 0.0, 0.145]), D1),
        (4.2, np.array([0.0, 0.0, 0.135]), L2),
        (3.0, np.array([0.04, 0.0, 0.045]), A3 + 0.12),
        (2.2, np.array([0.151, 0.0, 0.0]), D4),
        (1.0, np.array([0.025, 0.0, 0.0]), D6),
        (0.3 + float(payload_kg), np.array([0.036, 0.0, 0.0]), D6),
    ]
    links = []
    for m, com, L in specs:
        # Inercia de cuerpo rigido equivalente (varilla + radio transversal ~0.04 m)
        r_eq = 0.04
        I_rod = (m * (L * L) / 12.0 + 0.25 * m * r_eq * r_eq)
        I = np.diag([I_rod, I_rod, I_rod]).astype(float)
        links.append({'m': float(m), 'com': com.copy(), 'I': I})
    return links


def rne_real(q_rad, qd_rad, qdd_rad, params=None, gvec=GRAVITY):
    """
    Newton-Euler Recursivo (RNE) exacto para REAL_CHAIN:
      tau = M(q) qdd + C(q,qd) qd + g(q)   [N*m]
    """
    if params is None:
        params = irb120_ref_dyn_params()
    n = 6
    q = np.asarray(q_rad, dtype=float)
    qd = np.asarray(qd_rad, dtype=float)
    qdd = np.asarray(qdd_rad, dtype=float)

    R = [_ROT_FN[REAL_CHAIN[i][1]](float(q[i])) for i in range(n)]
    p = [REAL_CHAIN[i][0] for i in range(n)]
    ax = [_AXES[REAL_CHAIN[i][1]] for i in range(n)]

    w = np.zeros(3)
    wd = np.zeros(3)
    vd = -np.asarray(gvec, dtype=float)
    F, N = [], []

    for i in range(n):
        Rt = R[i].T
        w_p = Rt @ w
        wd_new = Rt @ wd + ax[i] * qdd[i] + _cross3(w_p, ax[i] * qd[i])
        vd = Rt @ (vd + _cross3(wd, p[i]) + _cross3(w, _cross3(w, p[i])))
        w = w_p + ax[i] * qd[i]
        wd = wd_new
        c, m, I = params[i]['com'], params[i]['m'], params[i]['I']
        a_c = vd + _cross3(wd, c) + _cross3(w, _cross3(w, c))
        F.append(m * a_c)
        N.append(I @ wd + _cross3(w, I @ w))

    tau = np.zeros(n)
    f_next = np.zeros(3)
    n_next = np.zeros(3)
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            fc = R[i + 1] @ f_next
            nc = R[i + 1] @ n_next
            pn = p[i + 1]
        else:
            fc = np.zeros(3)
            nc = np.zeros(3)
            pn = np.zeros(3)
        c = params[i]['com']
        f = fc + F[i]
        nn = N[i] + nc + _cross3(c, F[i]) + _cross3(pn, fc)
        tau[i] = float(nn @ ax[i])
        f_next, n_next = f, nn
    return tau


def mass_matrix_real(q_rad, params=None):
    if params is None:
        params = irb120_ref_dyn_params()
    M = np.zeros((6, 6), dtype=float)
    z = np.zeros(6)
    g0 = np.zeros(3)
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1.0
        M[:, i] = rne_real(q_rad, z, e, params=params, gvec=g0)
    return M


def gravity_torque_real(q_rad, params=None):
    if params is None:
        params = irb120_ref_dyn_params()
    z = np.zeros(6)
    return rne_real(q_rad, z, z, params=params, gvec=GRAVITY)


def coriolis_torque_real(q_rad, qd_rad, params=None):
    if params is None:
        params = irb120_ref_dyn_params()
    return rne_real(q_rad, qd_rad, np.zeros(6), params=params, gvec=np.zeros(3))


# =============================================================================
# 5. EVALUACION INTEGRAL HITO 1 + HITO 2 Y FORMATEO DE REPORTE
# =============================================================================
def evaluar_hito2(
    q_target_deg,
    q_current_deg=None,
    v_tcp_mms=30.0,
    v_ori_degs=15.0,
    amax_degs2=30.0,
    payload_kg=0.0,
):
    """
    Evalua Hito 1 + todo Hito 2 (IK Pieper, LSPB, Jacobiano 6x6 y Dinamica RNE)
    para el movimiento desde q_current_deg hasta q_target_deg.
    """
    q_tgt_deg = [float(v) for v in q_target_deg]
    q_cur_deg = [float(v) for v in (q_current_deg if q_current_deg is not None else q_tgt_deg)]
    q_tgt_rad = np.radians(q_tgt_deg)
    q_cur_rad = np.radians(q_cur_deg)

    # Validar rangos de velocidad tal como el usuario los solicito
    speed_in_range = (
        SPEED_TCP_RANGE[0] <= float(v_tcp_mms) <= SPEED_TCP_RANGE[1]
        and SPEED_ORI_RANGE[0] <= float(v_ori_degs) <= SPEED_ORI_RANGE[1]
        and AMAX_RANGE_DEG[0] <= float(amax_degs2) <= AMAX_RANGE_DEG[1]
    )

    h1 = evaluar_seguridad_hito1(q_tgt_deg, q_current_deg=q_current_deg)
    T_tgt = h1['fk_target']['T0_6']

    # 1) IK Pieper desde pose cartesiana objetivo
    ik_res = ik_pieper_todas(T_tgt, q_ref_rad=q_tgt_rad)

    # 2) LSPB sincronizada (muestras clave + transiciones exactas)
    spd_cfg = velocidades_lspb_desde_usuario(v_tcp_mms, v_ori_degs, amax_degs2)
    traj = LSPBReal(q_cur_rad, q_tgt_rad, spd_cfg['vmax_rad'], spd_cfg['amax_rad'])
    t_arr, Q_arr, Qd_arr, Qdd_arr = traj.sample_keyframes(n_steps=21)

    # 3) Jacobiano en destino y barrido cartesiano a lo largo de LSPB
    jac_tgt = analizar_jacobiano_rad(q_tgt_rad)
    jac_cur = analizar_jacobiano_rad(q_cur_rad)
    v_cart_mms_hist = []
    w_cart_degs_hist = []
    sigma_min_traj = []
    for qi, qdi in zip(Q_arr, Qd_arr):
        Ji = jacobiano_geometrico_rad(qi)
        xi = Ji @ qdi
        v_cart_mms_hist.append(float(np.linalg.norm(xi[:3]) * 1000.0))
        w_cart_degs_hist.append(float(math.degrees(np.linalg.norm(xi[3:]))))
        sigma_min_traj.append(float(np.linalg.svd(Ji, compute_uv=False)[-1]))

    # 4) Dinamica RNE en destino y a lo largo de la trayectoria
    dyn_params = irb120_ref_dyn_params(payload_kg=payload_kg)
    g_tgt = gravity_torque_real(q_tgt_rad, params=dyn_params)
    M_tgt = mass_matrix_real(q_tgt_rad, params=dyn_params)
    tau_hist = np.zeros_like(Q_arr)
    g_hist = np.zeros_like(Q_arr)
    c_hist = np.zeros_like(Q_arr)
    iner_hist = np.zeros_like(Q_arr)
    zeros6 = np.zeros(6)
    zeros3 = np.zeros(3)
    for k, (qk, qdk, qddk) in enumerate(zip(Q_arr, Qd_arr, Qdd_arr)):
        tau_k = rne_real(qk, qdk, qddk, params=dyn_params, gvec=GRAVITY)
        g_k = rne_real(qk, zeros6, zeros6, params=dyn_params, gvec=GRAVITY)
        c_k = rne_real(qk, qdk, zeros6, params=dyn_params, gvec=zeros3)
        tau_hist[k] = tau_k
        g_hist[k] = g_k
        c_hist[k] = c_k
        iner_hist[k] = tau_k - g_k - c_k

    tau_peak = np.max(np.abs(tau_hist), axis=0)
    g_peak = np.max(np.abs(g_hist), axis=0)
    c_peak = np.max(np.abs(c_hist), axis=0)
    iner_peak = np.max(np.abs(iner_hist), axis=0)
    qd_peak_deg = np.degrees(np.max(np.abs(Qd_arr), axis=0))
    qdd_peak_deg = np.degrees(np.max(np.abs(Qdd_arr), axis=0))
    torque_ok = bool(np.all(tau_peak <= JOINT_TAU_REF_NM))

    # Checks especificos Hito 2
    checks_h2 = []
    checks_h2.append({
        'nombre': '5. Velocidades seguras de usuario',
        'ok': speed_in_range,
        'detalle': (
            f'v_tcp={float(v_tcp_mms):.1f} mm/s [5..150], '
            f'v_ori={float(v_ori_degs):.1f} °/s [5..60], '
            f'a_max={float(amax_degs2):.1f} °/s² [5..180]'
        ),
    })
    ik_chosen = ik_res['chosen']
    ik_ok = ik_chosen is not None and ik_chosen['err_pos_mm'] < 0.1
    checks_h2.append({
        'nombre': '6. IK Pieper (desacople y consistencia)',
        'ok': ik_ok,
        'detalle': (
            f"{len(ik_res['sols'])} rama(s) valida(s); error FK(IK)={ik_chosen['err_pos_mm']:.4f} mm / {ik_chosen['err_ori_deg']:.4f}°"
            if ik_chosen else 'Sin solucion IK dentro de limites'
        ),
    })
    # Nota informativa de singularidades (no bloquea MoveAbsJ articular, pero alerta!)
    sing_txt = 'Sin alertas en destino' if not jac_tgt['alertas'] else ' ; '.join(jac_tgt['alertas'])
    checks_h2.append({
        'nombre': '7. Jacobiano y singularidades (informativo)',
        'ok': True,
        'detalle': f'sigma_min={jac_tgt["sigma_min"]:.4f} | det(J)={jac_tgt["det"]:.5f} | {sing_txt}',
    })
    checks_h2.append({
        'nombre': '8. Pares dinamicos RNE dentro de limites',
        'ok': torque_ok,
        'detalle': 'Pico |tau|=[' + ', '.join(f'{v:.2f}' for v in tau_peak) + '] N·m',
    })

    all_checks = list(h1['checks']) + checks_h2
    aprobado = bool(h1['aprobado'] and all(c['ok'] for c in checks_h2))

    return {
        'aprobado': aprobado,
        'hito1': h1,
        'checks': all_checks,
        'checks_h2': checks_h2,
        'speed_cfg': spd_cfg,
        'ik': ik_res,
        'lspb': {
            'traj': traj,
            'T': traj.T,
            'tb': traj.tb,
            'D': traj.D,
            'V': traj.V,
            'A': traj.A,
            'vr': traj.vr,
            'ar': traj.ar,
            'tipo': traj.tipo,
            'junta_critica_v': traj.junta_critica_v,
            'junta_critica_a': traj.junta_critica_a,
            't': t_arr,
            'Q': Q_arr,
            'Qd': Qd_arr,
            'Qdd': Qdd_arr,
            'qd_peak_deg': qd_peak_deg,
            'qdd_peak_deg': qdd_peak_deg,
        },
        'jac': {
            'target': jac_tgt,
            'current': jac_cur,
            'v_cart_max_mms': max(v_cart_mms_hist) if v_cart_mms_hist else 0.0,
            'w_cart_max_degs': max(w_cart_degs_hist) if w_cart_degs_hist else 0.0,
            'sigma_min_traj': min(sigma_min_traj) if sigma_min_traj else jac_tgt['sigma_min'],
        },
        'dyn': {
            'params': dyn_params,
            'M_target': M_tgt,
            'g_target': g_tgt,
            'tau_peak': tau_peak,
            'g_peak': g_peak,
            'c_peak': c_peak,
            'iner_peak': iner_peak,
            'tau_hist': tau_hist,
            'total_moving_mass_kg': sum(d['m'] for d in dyn_params),
        },
    }


def _fmt_vec(vec, unit='', width=7, dec=2):
    u = f' {unit}' if unit else ''
    return '[' + ', '.join(f'{float(x):{width}.{dec}f}' for x in vec) + ']' + u


def _fmt_mat(M, width=8, dec=4):
    return ['  [ ' + '  '.join(f'{float(v):{width}.{dec}f}' for v in row) + ' ]' for row in M]


def formatear_reporte_hito2(res, modo='hito2', detallado=False, robtarget_real_mm=None):
    """
    Formatea las ecuaciones y valores evaluados para:
      modo = 'hito1' -> Reporte clasico de Hito 1
      modo = 'hito2' -> Todas las ecuaciones del Hito 2 (IK, LSPB, Jacobiano, RNE)
      modo = 'todo'  -> Hito 1 + Hito 2 completos
    """
    if modo == 'hito1':
        return formatear_reporte_hito1(res['hito1'], robtarget_real_mm=robtarget_real_mm, paso_a_paso=detallado)

    h1 = res['hito1']
    spd = res['speed_cfg']
    ik = res['ik']
    lspb = res['lspb']
    jac = res['jac']
    jt = jac['target']
    dyn = res['dyn']

    lines = []
    lines.append('=' * 76)
    titulo = 'HITO 1 + HITO 2 COMPLETO' if modo == 'todo' else 'HITO 2 EN ROBOT REAL'
    lines.append(f'  {titulo} - VELOCIDADES, IK PIEPER, LSPB, JACOBIANO Y DINAMICA RNE')
    lines.append('=' * 76)

    if modo == 'todo':
        lines.append(formatear_reporte_hito1(h1, robtarget_real_mm=robtarget_real_mm, paso_a_paso=detallado))
        lines.append('')

    pt = h1['fk_target']['pos_mm']
    r_d, p_d, y_d = h1['fk_target']['rpy_deg']
    lines.append(f'Pose objetivo: X={pt[0]:.2f} Y={pt[1]:.2f} Z={pt[2]:.2f} mm | RPY=({r_d:.1f}, {p_d:.1f}, {y_d:.1f})°')
    lines.append(
        f'Velocidad seleccionada: v_tcp = {spd["v_tcp_mms"]:.1f} mm/s | '
        f'v_ori = {spd["v_ori_degs"]:.1f} °/s | a_max = {spd["amax_degs2"]:.1f} °/s²'
    )

    # --- 1) IK Pieper ---
    lines.append('')
    lines.append('--- [A] CINEMATICA INVERSA ANALITICA (METODO DE PIEPER) ---')
    lines.append('  Ecuaciones:')
    lines.append('    p_w = p_des - d6 * x_6     (centro de muneca; d6 = 0.072 m)')
    lines.append('    r = cos(q1)*p_wx + sin(q1)*p_wy ;  z = p_wz - d1   (d1 = 0.290 m)')
    lines.append('    cos(phi) = (r^2 + z^2 - L2^2 - L3^2) / (2*L2*L3)   (L2=0.270, L3=0.3100 m)')
    lines.append('    q2 = atan2(r,z) - atan2(L3*sin(phi), L2+L3*cos(phi)) ; q3 = phi - alpha0')
    lines.append('    R_w = (R_0^3)^T * R_des = Rx(q4)*Ry(q5)*Rx(q6)  =>  (q4, q5, q6)')
    pw = ik['p_wc_mm']
    lines.append(f'  Evaluacion: p_w = ({pw[0]:.2f}, {pw[1]:.2f}, {pw[2]:.2f}) mm | ramas validas = {len(ik["sols"])} / 8')
    ch = ik['chosen']
    if ch is not None:
        lines.append(
            f'    Rama elegida: {ch["branch"][0]} / {ch["branch"][1]} / {ch["branch"][2]} '
            f'(err FK={ch["err_pos_mm"]:.4f} mm, {ch["err_ori_deg"]:.4f}°)'
        )
        lines.append('    q_IK = ' + _fmt_vec(ch['q_deg'], '°', width=6, dec=2))
        if detallado:
            for idx, sol in enumerate(ik['sols'], start=1):
                sg = ' [SING]' if sol['singular'] else ''
                lines.append(f'      Rama {idx}: {sol["branch"]} -> ' + _fmt_vec(sol['q_deg'], '°', 6, 2) + sg)
    else:
        lines.append('    SIN SOLUCION IK DENTRO DE LIMITES ARTICULARES')

    # --- 2) LSPB ---
    lines.append('')
    lines.append('--- [B] TRAYECTORIA LSPB SINCRONIZADA (q(t), qd(t), qdd(t)) ---')
    lines.append('  Ecuaciones:')
    lines.append('    q_i(t) = q0_i + s(t)*(qf_i - q0_i) ;  qd = s\'(t)*dq ;  qdd = s\'\'(t)*dq')
    lines.append('    vr = max|dq_i|/vmax_i ;  ar = max|dq_i|/amax_i ;  D = max(vr, sqrt(ar))')
    lines.append('    tb = ar / D ;  T = D + tb ;  V = 1 / D ;  A = 1 / (tb * D)')
    lines.append('  Evaluacion con limites actuales:')
    lines.append('    vmax = ' + _fmt_vec(spd['vmax_deg'], '°/s', width=5, dec=1))
    lines.append(
        f'    Perfil: {lspb["tipo"].upper()} | T={lspb["T"]:.3f} s | tb={lspb["tb"]:.3f} s | '
        f'D={lspb["D"]:.3f} s (critica J{lspb["junta_critica_v"]})'
    )
    lines.append('    |qd|_max  = ' + _fmt_vec(lspb['qd_peak_deg'], '°/s', width=5, dec=2))
    lines.append('    |qdd|_max = ' + _fmt_vec(lspb['qdd_peak_deg'], '°/s²', width=5, dec=1))
    if detallado and lspb['T'] > 0.0:
        lines.append('    Muestras clave (t [s] -> s, q [°]):')
        for tk in [0.0, lspb['tb'], 0.5 * lspb['T'], lspb['T'] - lspb['tb'], lspb['T']]:
            qk, _, _ = lspb['traj'].at(tk)
            sk, sdk, _ = lspb['traj'].scalar_at(tk)
            lines.append(f'      t={tk:5.2f}s (s={sk:4.2f}, s\'={sdk:4.2f}) -> ' + _fmt_vec(np.degrees(qk), '°', 6, 1))

    # --- 3) Jacobiano ---
    lines.append('')
    lines.append('--- [C] JACOBIANO GEOMETRICO 6x6 Y SINGULARIDADES ---')
    lines.append('  Ecuaciones:')
    lines.append('    xi = [v; w]^T = J(q) * qd    con  J_v,i = z_i x (p_e - p_i),  J_w,i = z_i')
    lines.append(
        f'  Evaluacion en destino: det(J)={jt["det"]:.5f} | sigma_min={jt["sigma_min"]:.4f} | '
        f'cond(J)={jt["cond"]:.1f}' if math.isfinite(jt['cond']) else
        f'  Evaluacion en destino: det(J)={jt["det"]:.5f} | sigma_min={jt["sigma_min"]:.4f} | cond(J)=inf (singular)'
    )
    lines.append(
        f'    |sin(q5)|={jt["sin_muneca"]:.4f} | |sin(q3+a0)|={jt["sin_codo"]:.4f} | '
        f'rho_muneca={jt["rho_hombro_mm"]:.1f} mm'
    )
    lines.append(
        f'    Vel. cartesiana pico en trayectoria: ||v||_max={jac["v_cart_max_mms"]:.2f} mm/s | '
        f'||w||_max={jac["w_cart_max_degs"]:.2f} °/s | min(sigma_min)={jac["sigma_min_traj"]:.4f}'
    )
    if jt['alertas']:
        for al in jt['alertas']:
            lines.append('    ALERTA: ' + al)
    if detallado:
        lines.append('  Matriz J(q_destino) [filas 1-3: m/rad, filas 4-6: adim]:')
        lines.extend(_fmt_mat(jt['J'], width=7, dec=4))

    # --- 4) Dinamica RNE ---
    lines.append('')
    lines.append('--- [D] DINAMICA INVERSA NEWTON-EULER RECURSIVO (RNE) ---')
    lines.append('  Ecuacion:')
    lines.append('    tau = M(q)*qdd + C(q,qd)*qd + g(q)    [N·m]  (masa movil ref = '
                 f'{dyn["total_moving_mass_kg"]:.1f} kg)')
    lines.append('  Evaluacion:')
    lines.append('    g(q_dest)  = ' + _fmt_vec(dyn['g_target'], 'N·m', width=6, dec=3))
    lines.append('    |tau|_max  = ' + _fmt_vec(dyn['tau_peak'], 'N·m', width=6, dec=3))
    lines.append('    |M*qdd|_mx = ' + _fmt_vec(dyn['iner_peak'], 'N·m', width=6, dec=3))
    lines.append('    |C*qd|_mx  = ' + _fmt_vec(dyn['c_peak'], 'N·m', width=6, dec=3))
    if detallado:
        lines.append('  Matriz de Inercia M(q_dest) [kg·m^2]:')
        lines.extend(_fmt_mat(dyn['M_target'], width=7, dec=4))

    # --- 5) Seguridad ---
    lines.append('')
    lines.append('--- VERIFICACION DE SEGURIDAD (HITO 1 + HITO 2) ---')
    for chk in res['checks']:
        tag = 'OK' if chk['ok'] else 'BLOQUEADO'
        lines.append(f'  [{tag}] {chk["nombre"]}: {chk["detalle"]}')

    lines.append('-' * 76)
    if res['aprobado']:
        lines.append('  VEREDICTO: SEGURO Y VALIDADO (HITO 1 + HITO 2) — LISTO PARA PEDIR PERMISO')
    else:
        lines.append('  VEREDICTO: MOVIMIENTO BLOQUEADO POR SEGURIDAD (NO SE ENVIARA)')
    lines.append('=' * 76)
    return '\n'.join(lines)


# =============================================================================
# AUTOPRUEBAS NUMERICAS INDEPENDIENTES
# =============================================================================
def _run_self_tests():
    rng = np.random.default_rng(2026)
    print('=== Autopruebas matematicas Hito 2 (modelo real ABB IRB 120-3/0.6) ===')

    # 1. FK vs IK Pieper en 100 poses aleatorias dentro de limites
    max_ik_mm = 0.0
    max_ik_deg = 0.0
    for _ in range(100):
        q_rand = np.array([rng.uniform(lo * 0.85, hi * 0.85) for lo, hi in JOINT_LIMITS_DEG])
        q_rad = np.radians(q_rand)
        T = fk_real_rad(q_rad)
        ik = ik_pieper_todas(T, q_ref_rad=q_rad)
        assert ik['alcanzable'], 'Pose FK valida deberia tener solucion IK'
        ch = ik['chosen']
        max_ik_mm = max(max_ik_mm, ch['err_pos_mm'])
        max_ik_deg = max(max_ik_deg, ch['err_ori_deg'])
    assert max_ik_mm < 1e-6 and max_ik_deg < 1e-6
    print(f'[1/6] IK Pieper 100 poses: err_pos_max={max_ik_mm:.2e} mm, err_ori_max={max_ik_deg:.2e} deg -> OK')

    # 2. Jacobiano geometrico vs diferencias finitas centrales
    h = 1e-6
    max_j_err = 0.0
    for _ in range(50):
        q = np.radians([rng.uniform(lo * 0.8, hi * 0.8) for lo, hi in JOINT_LIMITS_DEG])
        J_ana = jacobiano_geometrico_rad(q)
        J_num = np.zeros((6, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = h
            Tp = fk_real_rad(q + dq)
            Tm = fk_real_rad(q - dq)
            J_num[:3, j] = (Tp[:3, 3] - Tm[:3, 3]) / (2.0 * h)
            R0 = fk_real_rad(q)[:3, :3]
            dR = (Tp[:3, :3] - Tm[:3, :3]) @ R0.T / (2.0 * h)
            J_num[3:, j] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) * 0.5
        max_j_err = max(max_j_err, float(np.max(np.abs(J_ana - J_num))))
    assert max_j_err < 1e-7
    print(f'[2/6] Jacobiano 6x6 vs diff finitas (50 poses): err_max={max_j_err:.2e} -> OK')

    # 3. RNE gravedad g(q) vs dU/dq y M(q) SPD + consistencia lineal
    params = irb120_ref_dyn_params()

    def potential_energy(q):
        U = 0.0
        T = np.eye(4)
        for i, (xyz, axis) in enumerate(REAL_CHAIN):
            T = T @ homog(_ROT_FN[axis](float(q[i])), xyz)
            p_c = T[:3, 3] + T[:3, :3] @ params[i]['com']
            U += -params[i]['m'] * float(np.dot(GRAVITY, p_c))
        return U

    max_g_err = 0.0
    max_rne_err = 0.0
    for _ in range(40):
        q = np.radians([rng.uniform(lo * 0.7, hi * 0.7) for lo, hi in JOINT_LIMITS_DEG])
        qd = rng.uniform(-2.0, 2.0, size=6)
        qdd = rng.uniform(-3.0, 3.0, size=6)
        g_ana = gravity_torque_real(q, params=params)
        g_num = np.zeros(6)
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = h
            g_num[j] = (potential_energy(q + dq) - potential_energy(q - dq)) / (2.0 * h)
        max_g_err = max(max_g_err, float(np.max(np.abs(g_ana - g_num))))

        M = mass_matrix_real(q, params=params)
        assert np.allclose(M, M.T, atol=1e-10), 'M debe ser simetrica'
        assert np.all(np.linalg.eigvalsh(M) > 0.0), 'M debe ser definida positiva'
        tau_direct = rne_real(q, qd, qdd, params=params)
        tau_decomp = M @ qdd + coriolis_torque_real(q, qd, params=params) + g_ana
        max_rne_err = max(max_rne_err, float(np.max(np.abs(tau_direct - tau_decomp))))
    assert max_g_err < 1e-6 and max_rne_err < 1e-10
    print(f'[3/6] RNE g(q) vs dU/dq: err={max_g_err:.2e} N*m | RNE Mqdd+c+g err={max_rne_err:.2e} -> OK')

    # 4. LSPB respeta limites y perfiles trapezoidal/triangular
    spd_slow = velocidades_lspb_desde_usuario(30.0, 15.0, 30.0)
    spd_fast = velocidades_lspb_desde_usuario(150.0, 60.0, 120.0)
    tr_trap = LSPBReal(np.zeros(6), np.radians([30, 0, 0, 0, 0, 0]), spd_slow['vmax_rad'], spd_slow['amax_rad'])
    tr_tria = LSPBReal(np.zeros(6), np.radians([0, 0, 0, 0, 0, 2]), spd_fast['vmax_rad'], spd_fast['amax_rad'])
    assert tr_trap.tipo == 'trapezoidal' and tr_tria.tipo == 'triangular'
    _, Q, Qd, Qdd = tr_trap.sample(0.01)
    assert np.all(np.abs(Qd) <= spd_slow['vmax_rad'] + 1e-9)
    assert np.all(np.abs(Qdd) <= spd_slow['amax_rad'] + 1e-9)
    tr_fast = LSPBReal(np.zeros(6), np.radians([30, 0, 0, 0, 0, 0]), spd_fast['vmax_rad'], spd_fast['amax_rad'])
    assert tr_fast.T < tr_trap.T
    print(f'[4/6] LSPB trap T={tr_trap.T:.3f}s (30 mm/s) vs fast T={tr_fast.T:.3f}s ({spd_fast["v_tcp_mms"]:.0f} mm/s), tria tb={tr_tria.tb:.3f}s -> OK')

    # 5. Evaluacion segura (home y pose valida) vs bloqueos (Z baja, velocidad alta, limite J1)
    ev_ok = evaluar_hito2([15.0, 10.0, -10.0, 0.0, 25.0, 10.0], [0.0] * 6, v_tcp_mms=60.0, v_ori_degs=25.0, amax_degs2=45.0)
    assert ev_ok['aprobado']
    ev_home = evaluar_hito2([0.0] * 6, [5.0, 0.0, 0.0, 0.0, 10.0, 0.0], v_tcp_mms=30.0, v_ori_degs=15.0, amax_degs2=30.0)
    assert ev_home['aprobado'], 'Regreso a Home con q5=0 debe permitirse con alerta informativa'
    ev_bad_z = evaluar_hito2([0.0, 85.0, 65.0, 0.0, 30.0, 0.0], [0.0] * 6)
    assert not ev_bad_z['aprobado']
    ev_bad_v = evaluar_hito2([10.0, 0.0, 0.0, 0.0, 20.0, 0.0], [0.0] * 6, v_tcp_mms=200.0)
    assert not ev_bad_v['aprobado']
    ev_bad_j = evaluar_hito2([170.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6)
    assert not ev_bad_j['aprobado']
    print('[5/6] Seguridad Hito 2: poses seguras + home aprobadas; Z baja / vel 200 / J1=170 bloqueadas -> OK')

    # 6. Formateo de reporte completo
    rep = formatear_reporte_hito2(ev_ok, modo='todo', detallado=True)
    assert 'CINEMATICA INVERSA ANALITICA' in rep and 'NEWTON-EULER RECURSIVO' in rep
    print('[6/6] Reporte completo Hito 1 + Hito 2 formateado (' + str(len(rep)) + ' chars) -> OK')
    print('TODAS LAS AUTOPRUEBAS DE HITO 2 PASARON!')


if __name__ == '__main__':
    _run_self_tests()
