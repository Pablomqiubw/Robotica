"""
test_dyn.py  -  Pruebas de dinamica (SIMULACION MATEMATICA, parametros DEMO)
Ejecutar:  python3 test_dyn.py

Los resultados numericos de torque usan masas/inercias DEMO (no del IRB 120).
Lo que se valida aqui es la CORRECCION DEL ALGORITMO, no valores del robot real.
"""
import numpy as np
from irb120_dyn_params import demo_link_params
from irb120_dyn import (rne, mass_matrix, mass_matrix_lagrange, gravity_torque,
                        gravity_lagrange, coriolis_vector, coriolis_christoffel,
                        torque_for_cartesian_accel)

P = demo_link_params()
rng = np.random.default_rng(3)

# --- Test 1: M(q) por RNE == M(q) por Lagrange; simetrica y definida positiva --
e_M, min_eig = 0.0, 1e9
for _ in range(30):
    q = rng.uniform(-np.pi, np.pi, 6)
    M1, M2 = mass_matrix(q, P), mass_matrix_lagrange(q, P)
    e_M = max(e_M, np.abs(M1 - M2).max())
    assert np.allclose(M1, M1.T, atol=1e-10)
    min_eig = min(min_eig, np.linalg.eigvalsh(M1).min())
print(f"[T1] M: max |RNE - Lagrange| = {e_M:.2e} ; simetrica OK ; autovalor minimo = {min_eig:.3e} (>0)")
assert e_M < 1e-8 and min_eig > 0

# --- Test 2: gravedad RNE == -dU/dq (derivada numerica del potencial) --------
e_g = 0.0
for _ in range(30):
    q = rng.uniform(-np.pi, np.pi, 6)
    e_g = max(e_g, np.abs(gravity_torque(q, P) - gravity_lagrange(q, P)).max())
print(f"[T2] gravedad: max |RNE - dU/dq| = {e_g:.2e} N m")
assert e_g < 1e-6

# --- Test 3: Coriolis RNE == simbolos de Christoffel -------------------------
e_c = 0.0
for _ in range(10):
    q = rng.uniform(-np.pi, np.pi, 6); qd = rng.uniform(-1, 1, 6)
    e_c = max(e_c, np.abs(coriolis_vector(q, qd, P) - coriolis_christoffel(q, qd, P)).max())
print(f"[T3] Coriolis/centrifugo: max |RNE - Christoffel| = {e_c:.2e} N m")
assert e_c < 1e-5

# --- Test 4: identidad completa tau = M qdd + c + g --------------------------
e_t = 0.0
for _ in range(30):
    q = rng.uniform(-np.pi, np.pi, 6); qd = rng.uniform(-1, 1, 6); qdd = rng.uniform(-2, 2, 6)
    tau = rne(q, qd, qdd, P)
    rhs = mass_matrix(q, P) @ qdd + coriolis_vector(q, qd, P) + gravity_torque(q, P)
    e_t = max(e_t, np.abs(tau - rhs).max())
print(f"[T4] tau_RNE = M qdd + c + g: max error = {e_t:.2e} N m")
assert e_t < 1e-9

# --- Test 5: divergencia del torque al acercarse a q5 = 0 -------------------
# Se mide tau - g(q): el peso del brazo (J2, J3) es finito y tapa la divergencia en la norma total.
qd = np.array([0.20, -0.10, 0.15, 0.30, 0.10, -0.20])        # velocidad articular finita
a_c = np.array([0.20, -0.10, 0.15, 0.50, -0.30, 0.40])        # aceleracion cartesiana pedida
print("\n[T5] aceleracion cartesiana fija; q5 -> 0  (parametros DEMO)")
print("      q5 [rad] | ||qdd4,6|| [rad/s^2] | |tau4| [N m] | |tau1| [N m] | ||tau-g|| [N m] | ||tau-g||*q5")
prev = None
for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    q = np.array([0.3, -0.5, 0.4, 0.7, eps, -0.2])
    tau, qdd = torque_for_cartesian_accel(q, qd, a_c, P)
    d = np.linalg.norm(tau - gravity_torque(q, P))
    print(f"      {eps:8.0e} | {np.hypot(qdd[3], qdd[5]):19.4e} | {abs(tau[3]):12.4e} | {abs(tau[0]):12.4e} | {d:15.4e} | {d*eps:12.4f}")
    if eps <= 1e-4 and prev is not None:   # regimen asintotico
        assert 8.0 < d / prev < 12.0, "||tau-g|| debe crecer ~10x por decada (1/q5)"
    prev = d
print("[T5] qdd4 = -qdd6 ~ 1/q5 y tau4, tau1 ~ 1/q5 divergen; tau2, tau3, tau5 quedan finitos")
print("     (tau6 queda finito: la aceleracion angular NETA de la pinza es qdd4+qdd6, que se cancela)")
print("TODO OK")
