"""
test_traj.py  -  Pruebas de LSPB  (SIMULACION MATEMATICA, no robot real)
Ejecutar:  python3 test_traj.py

IMPORTANTE: VMAX y AMAX de abajo son valores DEMO para probar el algoritmo.
NO son los limites del IRB 120. Sustituyelos por los del datasheet/URDF.
"""
import numpy as np
from irb120_traj import LSPB
from irb120_fk import forward_kinematics, pose_error
from irb120_ik import ik_pieper, closest_solution

VMAX = np.full(6, 1.0)     # rad/s   (DEMO)
AMAX = np.full(6, 2.0)     # rad/s^2 (DEMO)
TOL = 1e-9

# --- Test 1: caso trapezoidal (movimiento grande) ---------------------------
q0 = np.zeros(6)
qf = np.array([1.5, -1.0, 0.8, 2.0, -0.5, 0.3])
tr = LSPB(q0, qf, VMAX, AMAX)
t, Q, Qd, Qdd = tr.sample(1e-3)
print(f"[T1] trapezoidal: T={tr.T:.4f} s, tb={tr.tb:.4f} s, tramo lineal={tr.T-2*tr.tb:.4f} s")
assert tr.T - 2 * tr.tb > 0, "debia ser trapezoidal"
assert np.allclose(Q[0], q0, atol=TOL) and np.allclose(Q[-1], qf, atol=TOL)
assert np.allclose(Qd[0], 0, atol=TOL) and np.allclose(Qd[-1], 0, atol=TOL)
print("[T1] condiciones de borde (q y qd en t=0 y t=T): OK")

# --- Test 2: se respetan los limites y el limitante los alcanza -------------
assert np.all(np.abs(Qd).max(axis=0) <= VMAX + 1e-9)
assert np.all(np.abs(Qdd).max(axis=0) <= AMAX + 1e-9)
print(f"[T2] |qd| max por junta = {np.round(np.abs(Qd).max(axis=0), 4)}  (limite {VMAX[0]})")
print(f"[T2] |qdd| max por junta = {np.round(np.abs(Qdd).max(axis=0), 4)}  (limite {AMAX[0]})")
assert np.isclose(np.abs(Qd).max(), VMAX[0], atol=1e-9), "la junta limitante debe tocar vmax (tiempo minimo)"

# --- Test 3: derivada numerica de q coincide con qd analitica ---------------
qd_num = np.gradient(Q, t, axis=0)
err = np.abs(qd_num[2:-2] - Qd[2:-2]).max()
print(f"[T3] max |qd_numerica - qd_analitica| = {err:.2e} rad/s")
assert err < 1e-2   # diferencias finitas: error ~ dt * qdd en los cambios de tramo

# --- Test 4: caso triangular (movimiento pequeno) ---------------------------
tr2 = LSPB(q0, np.array([0.1, 0.05, -0.08, 0.02, 0.0, 0.03]), VMAX, AMAX)
t2, Q2, Qd2, _ = tr2.sample(1e-3)
print(f"[T4] triangular: T={tr2.T:.4f} s, tb={tr2.tb:.4f} s (tb = T/2: {np.isclose(tr2.tb, tr2.T/2)})")
assert np.isclose(tr2.tb, tr2.T / 2) and np.allclose(Q2[-1], tr2.qf, atol=TOL)

# --- Test 5: sin movimiento -------------------------------------------------
tr0 = LSPB(q0, q0, VMAX, AMAX)
assert tr0.T == 0.0 and np.allclose(tr0.at(0.0)[0], q0)
print("[T5] sin movimiento: T=0 OK")

# --- Test 6: sincronizacion (todas las juntas llegan a la vez, en linea recta)
s_i = (Q - q0) / np.where(qf - q0 == 0, 1, qf - q0)
assert np.allclose(s_i, s_i[:, [0]], atol=1e-9)
print("[T6] las 6 juntas siguen el mismo s(t): sincronizadas OK")

# --- Test 7: integracion IK -> LSPB -> FK -----------------------------------
q_start = np.array([0.2, -0.4, 0.3, 0.1, 0.6, -0.2])
q_goal_true = np.array([-0.5, 0.3, -0.2, 0.4, 0.9, 0.5])
T_goal = forward_kinematics(q_goal_true)             # "pose deseada"
sol = closest_solution(ik_pieper(T_goal), q_start)   # rama mas cercana (evita saltos)
tr3 = LSPB(q_start, sol['q'], VMAX, AMAX)
_, Q3, _, _ = tr3.sample(1e-2)
e_p, e_a = pose_error(forward_kinematics(Q3[-1]), T_goal)
print(f"[T7] IK->LSPB->FK: T={tr3.T:.3f} s, error final pos={e_p:.1e} m, ang={e_a:.1e} rad")
assert e_p < 1e-9 and e_a < 1e-9
print("TODO OK")
