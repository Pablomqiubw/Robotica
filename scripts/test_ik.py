"""
test_ik.py  -  Pruebas FK <-> IK  (SIMULACION MATEMATICA, no robot real)
Ejecutar:  python3 test_ik.py
"""
import numpy as np
from irb120_fk import forward_kinematics, pose_error
from irb120_ik import ik_pieper, L2, L3, D1, D_TOOL

rng = np.random.default_rng(42)
TOL_POS, TOL_ANG = 1e-9, 1e-9


def wrapdiff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


# --- Test 1: cada solucion devuelta debe reproducir la pose por FK ----------
max_pos = max_ang = 0.0
n_sol_hist, recovered, n_poses = {}, 0, 500
for _ in range(n_poses):
    q_true = rng.uniform(-np.pi, np.pi, 6)
    q_true[4] = rng.choice([-1, 1]) * rng.uniform(0.2, 2.5)   # lejos de singularidad
    T = forward_kinematics(q_true)
    sols = ik_pieper(T)
    n_sol_hist[len(sols)] = n_sol_hist.get(len(sols), 0) + 1
    for s in sols:
        e_p, e_a = pose_error(forward_kinematics(s['q']), T)
        max_pos, max_ang = max(max_pos, e_p), max(max_ang, e_a)
    if any(np.max(np.abs(wrapdiff(s['q'], q_true))) < 1e-6 for s in sols):
        recovered += 1
print(f"[T1] {n_poses} poses | max err pos = {max_pos:.2e} m | max err ang = {max_ang:.2e} rad")
print(f"[T1] q verdadero recuperado en {recovered}/{n_poses} | n soluciones: {n_sol_hist}")
assert max_pos < 1e-8 and max_ang < 1e-8 and recovered == n_poses

# --- Test 2: pose inalcanzable ----------------------------------------------
T_far = np.eye(4); T_far[:3, 3] = [5.0, 0, 0.3]
assert ik_pieper(T_far) == []
print("[T2] pose inalcanzable -> lista vacia  OK")

# --- Test 3: singularidad de muneca q5 = 0 ----------------------------------
q_s = np.array([0.3, -0.5, 0.4, 0.7, 0.0, -0.2])
T = forward_kinematics(q_s)
sols = ik_pieper(T)
sing = [x for x in sols if x['singular']]
assert sing, "debe detectarse al menos una solucion singular"
# TODAS las soluciones (singulares o no) deben reproducir la pose por FK
errs = [pose_error(forward_kinematics(x['q']), T) for x in sols]
assert max(e[0] for e in errs) < 1e-9 and max(e[1] for e in errs) < 1e-9
print(f"[T3] q5=0: {len(sols)} soluciones, {len(sing)} marcadas singulares; "
      f"max err pos={max(e[0] for e in errs):.1e}, ang={max(e[1] for e in errs):.1e}")
print(f"     q4+q6 verdadero = {q_s[3]+q_s[5]:.6f} | recuperado = {sing[0]['q'][3]+sing[0]['q'][5]:.6f}")
print("TODO OK")
print(f"\nConstantes derivadas del URDF: D1={D1}, L2={L2}, L3={L3:.4f}, D_TOOL={D_TOOL}")
