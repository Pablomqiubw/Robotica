"""
test_jac.py  -  Pruebas de Jacobiana y singularidad (SIMULACION MATEMATICA)
Ejecutar:  python3 test_jac.py
"""
import numpy as np
from irb120_fk import forward_kinematics, roty, rotz
from irb120_jac import (jacobian, joint_axes_and_origins, wrist_center,
                        singular_values, wrist_rates_analytic)
from irb120_params import JOINTS

rng = np.random.default_rng(7)


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# --- Test 1: Jacobiana analitica vs derivada numerica de la FK ---------------
max_err = 0.0
for _ in range(100):
    q = rng.uniform(-np.pi, np.pi, 6)
    J = jacobian(q)
    h = 1e-6
    Jn = np.zeros((6, 6))
    T0 = forward_kinematics(q)
    for i in range(6):
        e = np.zeros(6); e[i] = h
        Tp, Tm = forward_kinematics(q + e), forward_kinematics(q - e)
        Jn[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        Rdot = (Tp[:3, :3] - Tm[:3, :3]) / (2 * h)
        Jn[3:, i] = vee(Rdot @ T0[:3, :3].T)          # w = vee(Rdot R^T)
    max_err = max(max_err, np.abs(J - Jn).max())
print(f"[T1] 100 configs: max |J_analitica - J_numerica| = {max_err:.2e}")
assert max_err < 1e-7

# --- Test 2: det J = det(J11) * (z4 . (z5 x z6)),  |z4.(z5xz6)| = |sin q5| --
max_rel = 0.0
for _ in range(100):
    q = rng.uniform(-np.pi, np.pi, 6)
    zs, ps, _ = joint_axes_and_origins(q)
    pw = wrist_center(q)
    J11 = np.column_stack([np.cross(zs[i], pw - ps[i]) for i in range(3)])
    tri = zs[3] @ np.cross(zs[4], zs[5])
    assert abs(abs(tri) - abs(np.sin(q[4]))) < 1e-9
    d = np.linalg.det(jacobian(q))
    d_pred = np.linalg.det(J11) * tri
    max_rel = max(max_rel, abs(d - d_pred) / max(abs(d), 1e-12))
print(f"[T2] det(J) = det(J11)*sin(q5) (salvo signo): max error relativo = {max_rel:.2e}")
assert max_rel < 1e-8

# --- Test 3: divergencia de velocidades articulares cerca de q5 = 0 ---------
xi = np.array([0.10, -0.05, 0.08, 0.30, -0.20, 0.40])     # twist fijo [v; w] (m/s, rad/s)
print("\n[T3] twist fijo xi =", xi, "\n"
      "      eps=q5 [rad] |   det J    | sigma_min | cond(J)  | ||qd|| [rad/s] | ||qd||*eps")
prev = None
for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    q = np.array([0.3, -0.5, 0.4, 0.7, eps, -0.2])
    J = jacobian(q)
    s = singular_values(J)
    qd = np.linalg.solve(J, xi)
    n = np.linalg.norm(qd)
    print(f"      {eps:8.0e}      | {np.linalg.det(J):10.3e} | {s[-1]:9.2e} | {s[0]/s[-1]:8.2e} | {n:14.4e} | {n*eps:10.4f}")
    if prev is not None:
        assert 8.0 < n / prev < 12.0, "||qd|| debe crecer ~10x por cada decada (diverge como 1/q5)"
    prev = n
print("[T3] ||qd|| ~ 1/q5  y  sigma_min ~ q5  -> OK")

# --- Test 4: formulas analiticas de muneca coinciden con J^-1 ---------------
q = np.array([0.3, -0.5, 0.4, 0.7, 1e-3, -0.2])
J = jacobian(q)
qd = np.linalg.solve(J, xi)
zs, ps, _ = joint_axes_and_origins(q)
w_wrist = xi[3:] - sum(zs[i] * qd[i] for i in range(3))    # w de la muneca = w total - aporte J1..J3
# R_B = R(frame 3) * Ry(P4): orientacion del frame justo antes de J4
R_B = rotz(q[0]) @ roty(q[1] + q[2]) @ roty(JOINTS[3][1][1])
w_B = R_B.T @ w_wrist
qd_an = wrist_rates_analytic(w_B, q[3], q[4])
print(f"\n[T4] qd4..6 con J^-1      = {qd[3:]}")
print(f"[T4] qd4..6 analiticas    = {qd_an}")
assert np.allclose(qd[3:], qd_an, rtol=1e-6, atol=1e-6)
print("[T4] formulas analiticas qd4 ~ cot(q5), qd6 ~ 1/sin(q5): OK")

# --- Test 5: en q5 = 0 exacto la Jacobiana pierde rango ---------------------
q0 = np.array([0.3, -0.5, 0.4, 0.7, 0.0, -0.2])
J0 = jacobian(q0)
s0 = singular_values(J0)
print(f"\n[T5] q5=0: det J = {np.linalg.det(J0):.2e}, sigma_min = {s0[-1]:.2e}, rango = {np.linalg.matrix_rank(J0, tol=1e-9)}")
assert np.linalg.matrix_rank(J0, tol=1e-9) == 5
qd_pinv = np.linalg.pinv(J0, rcond=1e-9) @ xi
print(f"[T5] pinv da ||qd|| finito = {np.linalg.norm(qd_pinv):.3f} rad/s (aprox. minima norma, NO reproduce xi exacto)")
print("TODO OK")
