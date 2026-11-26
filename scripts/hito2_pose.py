import sys
import numpy as np
from irb120_fk import forward_kinematics, pose_error, rpy_to_R, homog
from irb120_ik import ik_pieper
from irb120_traj import LSPB
from irb120_jac import jacobian, singular_values
from irb120_dyn import rne, mass_matrix, gravity_torque, torque_for_cartesian_accel
from irb120_dyn_params import demo_link_params

np.set_printoptions(precision=4, suppress=True, linewidth=130)
wrap = lambda a: np.arctan2(np.sin(a), np.cos(a))
params = demo_link_params()
a = sys.argv[1:]

if len(a) == 7 and a[0] == 'pose':
    x, y, z, r, p, w = [float(v) for v in a[1:]]
    T = homog(rpy_to_R([r, p, w]), np.array([x, y, z]))
elif len(a) == 7 and a[0] == 'q':
    T = forward_kinematics(np.array([float(v) for v in a[1:]]))
else:
    T = forward_kinematics(np.array([0.4, -0.3, 0.5, 0.3, 0.8, -0.2]))
    print("(sin argumentos: uso una pose de ejemplo)")

print("\n######## POSE PEDIDA (pinza, en base_link) ########")
print(T)

print("\n######## T1: CINEMATICA INVERSA (Pieper) ########")
sols = ik_pieper(T)
print("Soluciones encontradas:", len(sols))
if not sols:
    print("Pose fuera del espacio de trabajo: no hay solucion. Prueba otra pose.")
    sys.exit(0)
for i, s in enumerate(sols, 1):
    e_p, e_a = pose_error(forward_kinematics(s['q']), T)
    print(f"{i}) rama={s['branch']} q={np.round(s['q'], 4)}  err_pos={e_p:.1e} m  err_ang={e_a:.1e} rad")

q0 = np.zeros(6)
best = min(sols, key=lambda s: np.linalg.norm(wrap(s['q'] - q0)))
qf = q0 + wrap(best['q'] - q0)
print("\nSolucion elegida (la mas cercana a q=0):", np.round(qf, 4))
print("Para mover RViz a esa misma pose:")
print("  rosrun irb120_description send_test_pose.py " + " ".join(f"{v:.4f}" for v in qf))

print("\n######## T2: TRAYECTORIA LSPB ########")
vmax, amax = np.full(6, 2.0), np.full(6, 5.0)
lspb = LSPB(q0, qf, vmax, amax)
t, Q, Qd, Qdd = lspb.sample(0.01)
print(f"T = {lspb.T:.3f} s, tb = {lspb.tb:.3f} s")
print("max|qd|  por junta =", np.abs(Qd).max(0))
print("max|qdd| por junta =", np.abs(Qdd).max(0))
e_p, e_a = pose_error(forward_kinematics(Q[-1]), T)
print(f"Error final de la trayectoria vs pose pedida: pos={e_p:.1e} m, ang={e_a:.1e} rad")

print("\n######## T3: JACOBIANO en la pose final ########")
J = jacobian(qf)
print(J)
sv = singular_values(J)
print("det J =", np.linalg.det(J))
print("valores singulares =", sv)
print("cond(J) =", sv[0] / sv[-1] if sv[-1] > 0 else np.inf, "| rango =", np.linalg.matrix_rank(J))
if sv[-1] < 1e-3:
    print("AVISO: cerca de una singularidad (sigma_min muy chico)")

print("\n######## T4: DINAMICA INVERSA (parametros DEMO) ########")
tau = np.array([rne(Q[i], Qd[i], Qdd[i], params) for i in range(len(t))])
print("max|tau| por junta a lo largo de la trayectoria [N m] =", np.abs(tau).max(0))
print("Torque de gravedad en la pose final g(q) [N m] =", gravity_torque(qf, params))
print("Matriz de masa M(q) en la pose final =")
print(mass_matrix(qf, params))

print("\n######## T5: SINGULARIDAD q5 -> 0 (desde la pose elegida) ########")
xi = np.array([0.05, 0.0, 0.0, 0.2, 0.1, 0.1])
a_c = np.array([0.1, 0.0, 0.0, 0.5, 0.3, 0.2])
print(f"{'q5 [rad]':>10} {'sigma_min':>11} {'||qd|| [rad/s]':>15} {'||tau-g|| [N m]':>16}")
for q5 in np.logspace(-0.5, -5, 8):
    q = qf.copy(); q[4] = q5
    Jq = jacobian(q)
    smin = singular_values(Jq)[-1]
    qd = np.linalg.solve(Jq, xi)
    tq, _ = torque_for_cartesian_accel(q, np.zeros(6), a_c, params)
    gq = gravity_torque(q, params)
    print(f"{q5:10.1e} {smin:11.2e} {np.linalg.norm(qd):15.3e} {np.linalg.norm(tq - gq):16.3e}")
print("Al acercarse q5 a 0: sigma_min -> 0 y velocidades y torques crecen ~ 1/q5.")
