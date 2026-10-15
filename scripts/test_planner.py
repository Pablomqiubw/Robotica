"""
test_planner.py  -  Pruebas del planificador y del nodo (SIMULACION MATEMATICA, sin robot real)
Ejecutar:  python3 test_planner.py     (no necesita ROS)

El URDF de abajo es un FIXTURE SINTETICO de prueba: tiene la geometria de irb120_params.py pero
sus limites (+-2.5 rad, velocity 2.0) son inventados solo para probar el codigo. NO es el IRB 120.
"""
import sys
import types
import numpy as np
from irb120_fk import forward_kinematics, pose_error
from irb120_params import JOINTS, GRIPPER_FIXED_XYZ
import irb120_planner as pl

rng = np.random.default_rng(11)
V, A = np.full(6, 2.0), np.full(6, 4.0)


def make_urdf(limits=True, vel=True):
    links = "".join(f'<link name="l{i}"/>' for i in range(8))
    js = ""
    for i, (xyz, rpy, ax) in enumerate(JOINTS):
        axis = {'x': '1 0 0', 'y': '0 1 0', 'z': '0 0 1'}[ax]
        lim = ''
        if limits:
            lim = f'<limit lower="-2.5" upper="2.5" effort="10"' + (' velocity="2.0"' if vel and i != 3 else '') + '/>'
        js += (f'<joint name="joint_{i+1}" type="revolute"><parent link="l{i}"/><child link="l{i+1}"/>'
               f'<origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/><axis xyz="{axis}"/>{lim}</joint>')
    js += '<joint name="gripper_fixed" type="fixed"><parent link="l6"/><child link="l7"/><origin xyz="0 0 0.072"/></joint>'
    return f'<robot name="synthetic_test">{links}{js}</robot>'


# --- T1: cuaterniones ida y vuelta ---------------------------------------------
worst = 0.0
for _ in range(200):
    R = forward_kinematics(rng.uniform(-np.pi, np.pi, 6))[:3, :3]
    worst = max(worst, np.abs(pl.quat_to_R(*pl.R_to_quat(R)) - R).max())
print(f"[T1] R -> cuaternion -> R: error max = {worst:.1e}")
assert worst < 1e-12

# --- T2: URDF: orden de cadena y limites (nada inventado) --------------------------
info = pl.parse_urdf_joints(make_urdf())
assert info['names'] == [f'joint_{i}' for i in range(1, 7)]
assert info['has_pos_limits'] and not info['has_vel_limits'] and np.isnan(info['velocity'][3])
info2 = pl.parse_urdf_joints(make_urdf(limits=False))
assert not info2['has_pos_limits'] and np.all(np.isinf(info2['lower']))
print("[T2] URDF: 6 juntas moviles en orden; lo que falta queda en nan/inf (no se inventa): OK")

# --- T3: plan a una pose alcanzable: rama valida, FK, trayectoria ---------------------
q_start = np.array([0.1, -0.2, 0.2, 0.1, 0.5, 0.0])
q_goal_true = np.array([0.6, -0.4, 0.5, 0.3, 0.9, -0.3])
T = forward_kinematics(q_goal_true)
res = pl.plan_to_pose(T, q_start, V, A, info['lower'], info['upper'])
assert res['ok'] and res['n_solutions'] == 8
print(f"[T3] plan OK: rama {res['branch']}, {res['n_valid']}/{res['n_solutions']} validas, T={res['lspb'].T:.3f} s, "
      f"err FK pos={res['fk_err_pos']:.1e} ang={res['fk_err_ang']:.1e}")
assert np.allclose(res['Q'][0], q_start) and np.allclose(res['Q'][-1], res['q_goal'])
assert np.allclose(res['q_goal'], q_goal_true, atol=1e-8), "debia elegir la rama mas cercana = la verdadera"
assert pose_error(forward_kinematics(res['Q'][-1]), T)[0] < 1e-9

# --- T4: pose inalcanzable ----------------------------------------------------------
Tf = np.eye(4); Tf[:3, 3] = [3.0, 0, 0.3]
r = pl.plan_to_pose(Tf, q_start, V, A)
assert not r['ok'] and 'espacio de trabajo' in r['reason']
print(f"[T4] inalcanzable -> ok=False: '{r['reason']}'")

# --- T5: limites articulares: si la rama cercana queda excluida, se elige otra ------------
wide_lo, wide_hi = np.full(6, -np.pi), np.full(6, np.pi)
r0 = pl.plan_to_pose(T, q_start, V, A, wide_lo, wide_hi)
assert r0['ok'] and r0['n_valid'] == 8
hi = wide_hi.copy(); hi[0] = 0.3                # q1 verdadero = 0.6 queda fuera -> debe usar la rama q1+pi
r = pl.plan_to_pose(T, q_start, V, A, wide_lo, hi)
print(f"[T5a] sin limites estrechos: {r0['n_valid']}/8 validas, rama {r0['branch']}, q1={r0['q_goal'][0]:.3f}")
print(f"[T5a] con q1<=0.3: ok={r['ok']}, validas={r['n_valid']}, rama {r['branch']}, q1={r['q_goal'][0]:.3f}, err FK={r['fk_err_pos']:.1e}")
assert r['ok'] and r['q_goal'][0] <= 0.3 + 1e-9 and r['fk_err_pos'] < 1e-9 and r['branch'][0] != r0['branch'][0] and r['n_valid'] < 8
lo2, hi2 = np.full(6, -0.01), np.full(6, 0.01)
r2 = pl.plan_to_pose(T, np.zeros(6), V, A, lo2, hi2)
assert not r2['ok']
print(f"[T5b] limites que excluyen todo -> ok=False: '{r2['reason']}'")

# --- T6: meta en singularidad q5 = 0 -----------------------------------------------
Ts = forward_kinematics(np.array([0.3, -0.5, 0.4, 0.7, 0.0, -0.2]))
r = pl.plan_to_pose(Ts, np.array([0.2, -0.4, 0.3, 0.5, 0.3, 0.0]), V, A, info['lower'], info['upper'])
assert r['ok'] and r['near_singularity'] and any('singularidad' in w for w in r['warnings'])
print(f"[T6] meta q5=0: ok, near_singularity=True, advertencia: '{r['warnings'][0][:60]}...'")

# --- T7: nodo ROS con rospy SIMULADO (valida el codigo del nodo, NO ROS real) ---------
class NS(types.SimpleNamespace):
    pass

clock = {'t': 0.0}
class FTime:
    def __init__(s, t): s.t = t
    def to_sec(s): return s.t
    def __sub__(s, o): return FTime(s.t - o.t)
    @staticmethod
    def now(): return FTime(clock['t'])
sent = {}
class FPub:
    def __init__(s, topic, *a, **k): s.topic = topic; sent[topic] = []
    def publish(s, m): sent[s.topic].append(m)
handlers = {}
params = {'robot_description': make_urdf(), '~amax': 4.0, '~vmax': 2.0}
fake_rospy = types.ModuleType('rospy')
fake_rospy.get_param = lambda n, d=None: params.get(n, d)
fake_rospy.Publisher = FPub
fake_rospy.Subscriber = lambda topic, cls, cb, **k: handlers.__setitem__(topic, cb)
fake_rospy.Timer = lambda d, cb: None
fake_rospy.Duration = lambda x: FTime(x)
fake_rospy.Time = FTime
logs = []
fake_rospy.loginfo = lambda m: logs.append(('I', m)); fake_rospy.logwarn = lambda m: logs.append(('W', m))
fake_rospy.logerr = lambda m: logs.append(('E', m))
msgs = {
 'geometry_msgs.msg': {'PoseStamped': lambda: NS(header=NS(frame_id='', stamp=None),
                       pose=NS(position=NS(x=0, y=0, z=0), orientation=NS(x=0, y=0, z=0, w=1)))},
 'sensor_msgs.msg': {'JointState': lambda: NS(header=NS(stamp=None), name=[], position=[], velocity=[])},
 'trajectory_msgs.msg': {'JointTrajectory': lambda: NS(header=NS(stamp=None), joint_names=[], points=[]),
                         'JointTrajectoryPoint': lambda: NS(positions=[], velocities=[], accelerations=[], time_from_start=None)},
 'std_msgs.msg': {'String': lambda data='': NS(data=data)}}
sys.modules['rospy'] = fake_rospy
for mod, attrs in msgs.items():
    m = types.ModuleType(mod); [setattr(m, k, v) for k, v in attrs.items()]
    sys.modules[mod] = m
    sys.modules[mod.split('.')[0]] = types.ModuleType(mod.split('.')[0])
import ik_lspb_node as node_mod

node = node_mod.IKLSPBNode()
pose_msg = msgs['geometry_msgs.msg']['PoseStamped']()
pose_msg.header.frame_id = 'base_link'
pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = T[:3, 3]
qq = pl.R_to_quat(T[:3, :3])
(pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w) = qq
node.q = q_start.copy()
handlers['irb120/target_pose'](pose_msg)
assert node.plan is not None, [m for k, m in logs if k == 'E']
assert len(sent['irb120/trajectory']) == 1 and sent['irb120/trajectory'][0].joint_names == info['names']
Tdur = node.plan['lspb'].T
n_js0 = len(sent['joint_states'])
for step in range(int(Tdur / 0.02) + 5):                     # reloj simulado a 50 Hz
    clock['t'] += 0.02
    node.on_timer(None)
last = sent['joint_states'][-1]
err = np.abs(np.array(last.position) - res['q_goal']).max()
print(f"[T7] nodo simulado: {len(sent['joint_states']) - n_js0} JointState, error final vs meta = {err:.1e} rad, "
      f"estado = '{sent['irb120/ik_status'][-1].data}'")
assert err < 1e-9 and node.plan is None and sent['irb120/ik_status'][-1].data.startswith('DONE')
node.q = q_start.copy()
bad = msgs['geometry_msgs.msg']['PoseStamped'](); bad.header.frame_id = 'world'
handlers['irb120/target_pose'](bad)
assert node.plan is None and sent['irb120/ik_status'][-1].data.startswith('FAIL')
print("[T7] frame distinto de base_link -> FAIL (sin TF): OK")
print("TODO OK")
