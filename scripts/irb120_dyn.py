"""
irb120_dyn.py
Dinamica inversa del brazo: tau = M(q) qdd + C(q,qd) qd + g(q)

Metodo principal: Newton-Euler recursivo (RNE), con la cadena tomada de JOINTS.
Metodo independiente de verificacion: Lagrange (M desde las Jacobianas).

Entradas : q, qd, qdd (6,) ; params = lista de 6 dicts {'m','com','I'} (ver irb120_dyn_params)
Salidas  : tau (6,) [N m]
"""
import numpy as np
from irb120_params import JOINTS
from irb120_fk import rotx, roty, rotz, rpy_to_R, homog
from irb120_jac import jacobian
from irb120_dyn_params import GRAVITY

_AXES = {'x': np.array([1.0, 0, 0]), 'y': np.array([0, 1.0, 0]), 'z': np.array([0, 0, 1.0])}
_ROT = {'x': rotx, 'y': roty, 'z': rotz}


def rne(q, qd, qdd, params, gvec=GRAVITY):
    """Newton-Euler recursivo. R[i] rota del marco del link i al de su padre; p[i] = origen de i en el padre."""
    n = 6
    R = [rpy_to_R(JOINTS[i][1]) @ _ROT[JOINTS[i][2]](q[i]) for i in range(n)]
    p = [np.array(JOINTS[i][0], dtype=float) for i in range(n)]
    ax = [_AXES[JOINTS[i][2]] for i in range(n)]

    w, wd, vd = np.zeros(3), np.zeros(3), -np.asarray(gvec, dtype=float)   # gravedad como aceleracion de la base
    F, N = [], []
    for i in range(n):                                      # --- pasada hacia adelante
        Rt = R[i].T
        w_p = Rt @ w
        wd_new = Rt @ wd + ax[i] * qdd[i] + np.cross(w_p, ax[i] * qd[i])
        vd = Rt @ (vd + np.cross(wd, p[i]) + np.cross(w, np.cross(w, p[i])))
        w = w_p + ax[i] * qd[i]
        wd = wd_new
        c, m, I = params[i]['com'], params[i]['m'], params[i]['I']
        a_c = vd + np.cross(wd, c) + np.cross(w, np.cross(w, c))
        F.append(m * a_c)
        N.append(I @ wd + np.cross(w, I @ w))

    tau = np.zeros(n)
    f_next, n_next = np.zeros(3), np.zeros(3)
    for i in range(n - 1, -1, -1):                          # --- pasada hacia atras
        if i < n - 1:
            fc, nc, pn = R[i + 1] @ f_next, R[i + 1] @ n_next, p[i + 1]
        else:
            fc, nc, pn = np.zeros(3), np.zeros(3), np.zeros(3)
        c = params[i]['com']
        f = fc + F[i]
        nn = N[i] + nc + np.cross(c, F[i]) + np.cross(pn, fc)
        tau[i] = nn @ ax[i]
        f_next, n_next = f, nn
    return tau


def gravity_torque(q, params):
    return rne(q, np.zeros(6), np.zeros(6), params)


def coriolis_vector(q, qd, params):
    """c(q,qd) = C(q,qd) qd  (sin gravedad)."""
    return rne(q, qd, np.zeros(6), params, gvec=np.zeros(3))


def mass_matrix(q, params):
    """M(q) columna a columna con RNE: tau(qdd = e_i, qd = 0, g = 0)."""
    M = np.zeros((6, 6))
    for i in range(6):
        e = np.zeros(6); e[i] = 1.0
        M[:, i] = rne(q, np.zeros(6), e, params, gvec=np.zeros(3))
    return M


# ---------------- Verificacion independiente (Lagrange) ----------------------
def _link_kinematics(q, params):
    T = np.eye(4)
    out = []
    for i, (xyz, rpy, axis) in enumerate(JOINTS):
        T = T @ homog(rpy_to_R(rpy), np.array(xyz, dtype=float)) @ homog(_ROT[axis](q[i]), np.zeros(3))
        out.append((T[:3, :3].copy(), T[:3, 3] + T[:3, :3] @ params[i]['com']))
    return out


def mass_matrix_lagrange(q, params):
    """M = sum_i [ m_i Jv_i^T Jv_i + Jw_i^T (R_i I_i R_i^T) Jw_i ]  con Jacobianas hasta la junta i."""
    M = np.zeros((6, 6))
    for i, (R0, pc) in enumerate(_link_kinematics(q, params)):
        J = jacobian(q, point=pc)
        J[:, i + 1:] = 0.0
        Jv, Jw = J[:3], J[3:]
        M += params[i]['m'] * Jv.T @ Jv + Jw.T @ (R0 @ params[i]['I'] @ R0.T) @ Jw
    return M


def gravity_lagrange(q, params, gvec=GRAVITY, h=1e-6):
    """g(q) = dU/dq con U = - sum m g.p_com (derivada numerica)."""
    def U(qq):
        return -sum(params[i]['m'] * gvec @ pc for i, (_, pc) in enumerate(_link_kinematics(qq, params)))
    g = np.zeros(6)
    for i in range(6):
        e = np.zeros(6); e[i] = h
        g[i] = (U(q + e) - U(q - e)) / (2 * h)
    return g


def coriolis_christoffel(q, qd, params, h=1e-6):
    """c_k = sum_ij Gamma_kij qd_i qd_j,  Gamma = 1/2 (dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k)."""
    dM = np.zeros((6, 6, 6))                                 # dM[l] = dM/dq_l
    for l in range(6):
        e = np.zeros(6); e[l] = h
        dM[l] = (mass_matrix_lagrange(q + e, params) - mass_matrix_lagrange(q - e, params)) / (2 * h)
    c = np.zeros(6)
    for k in range(6):
        for i in range(6):
            for j in range(6):
                c[k] += 0.5 * (dM[i][k, j] + dM[j][k, i] - dM[k][i, j]) * qd[i] * qd[j]
    return c


# ---------------- Torque para una aceleracion cartesiana ---------------------
def jdot_qd(q, qd, h=1e-6):
    """J_dot * qd por diferencia central de J a lo largo de qd."""
    return (jacobian(q + h * qd) - jacobian(q - h * qd)) @ qd / (2 * h)


def torque_for_cartesian_accel(q, qd, a_cart, params):
    """
    a_cart = [a_lineal ; alpha_angular] deseada en la pinza (base_link).
    qdd = J^-1 (a_cart - Jdot qd) ;  tau = M qdd + c + g.   (diverge cerca de q5 = 0)
    """
    J = jacobian(q)
    qdd = np.linalg.solve(J, a_cart - jdot_qd(q, qd))
    return rne(q, qd, qdd, params), qdd
