#!/usr/bin/env python3
"""
ik_lspb_node.py  -  Nodo ROS (Noetic): recibe una pose, ejecuta IK de Pieper y genera una trayectoria LSPB.

*** ES UNA SIMULACION: el nodo publica /joint_states para que RViz mueva el modelo URDF. ***
*** NO envia nada a un controlador ni al robot real. Para el robot real habria que usar el  ***
*** topic /irb120/trajectory con un controlador/driver del fabricante, con limites validados. ***

Topics
  Suscribe : ~target_topic (default irb120/target_pose)  geometry_msgs/PoseStamped
             pose de la PINZA (mismo frame que la FK del Hito 1) en base_link
  Publica  : joint_states               sensor_msgs/JointState        (solo si ~publish_joint_states)
             irb120/trajectory          trajectory_msgs/JointTrajectory (trayectoria completa)
             irb120/ik_status           std_msgs/String               (OK / FAIL + motivo)
Parametros (~)
  base_frame (base_link) | rate (50) | traj_dt (0.05) | vel_scale (1.0)
  robot_description_param (robot_description) | publish_joint_states (true)
  amax   : escalar o lista de 6 [rad/s^2]  -> NO esta en el URDF; default 2.0 = DEMO (se avisa)
  vmax   : opcional (escalar/lista); si no se da, se toma <limit velocity> del URDF
  vmax_default : usado SOLO si el URDF no declara velocity (default 1.0 = DEMO, se avisa)
  q_home : lista de 6 (default ceros)
"""
import os
import sys
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import irb120_planner as pl


def _vec6(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(6, float(a))
    if a.shape != (6,):
        raise ValueError(f"parametro {name} debe ser escalar o lista de 6")
    return a


class IKLSPBNode:
    def __init__(self):
        p = rospy.get_param
        self.base_frame = p('~base_frame', 'base_link')
        self.rate_hz = float(p('~rate', 50.0))
        self.traj_dt = float(p('~traj_dt', 0.05))
        self.publish_js = bool(p('~publish_joint_states', True))
        vel_scale = float(p('~vel_scale', 1.0))

        info = pl.parse_urdf_joints(p(p('~robot_description_param', 'robot_description')))
        if len(info['names']) != 6:
            raise RuntimeError(f"el URDF tiene {len(info['names'])} juntas moviles, se esperaban 6")
        self.names = info['names']
        self.lo, self.hi = info['lower'], info['upper']
        if not info['has_pos_limits']:
            rospy.logwarn("El URDF no declara limites articulares para todas las juntas: "
                          "no se filtran soluciones por limites (inf donde falta).")

        vmax_param = p('~vmax', None)
        if vmax_param is not None:
            self.vmax = _vec6(vmax_param, 'vmax')
        else:
            v = info['velocity'].copy()
            if np.any(np.isnan(v)):
                dflt = float(p('~vmax_default', 1.0))
                rospy.logwarn(f"El URDF no declara <limit velocity> en todas las juntas: se usa {dflt} rad/s "
                              f"(valor DEMO, NO es del IRB 120).")
                v[np.isnan(v)] = dflt
            self.vmax = v * vel_scale
        amax_param = p('~amax', None)
        if amax_param is None:
            rospy.logwarn("~amax no definido: se usa 2.0 rad/s^2 (valor DEMO, NO es del IRB 120).")
            amax_param = 2.0
        self.amax = _vec6(amax_param, 'amax')

        self.q = _vec6(p('~q_home', 0.0), 'q_home')
        if np.any(self.q < self.lo) or np.any(self.q > self.hi):
            rospy.logwarn("q_home esta fuera de los limites articulares del URDF.")

        self.plan = None
        self.t_start = None
        self.js_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.traj_pub = rospy.Publisher('irb120/trajectory', JointTrajectory, queue_size=1)
        self.status_pub = rospy.Publisher('irb120/ik_status', String, queue_size=10)
        rospy.Subscriber(p('~target_topic', 'irb120/target_pose'), PoseStamped, self.on_pose, queue_size=1)
        if not self.publish_js:
            rospy.Subscriber('joint_states', JointState, self.on_external_js, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.on_timer)
        rospy.loginfo(f"ik_lspb_node listo. Juntas: {self.names}")
        rospy.loginfo(f"vmax={np.round(self.vmax, 3)} rad/s | amax={np.round(self.amax, 3)} rad/s^2")

    # -- utilidades ---------------------------------------------------------
    def status(self, text, error=False):
        (rospy.logerr if error else rospy.loginfo)(text)
        self.status_pub.publish(String(data=text))

    def publish_state(self, q, qd=None):
        m = JointState()
        m.header.stamp = rospy.Time.now()
        m.name = list(self.names)
        m.position = [float(x) for x in q]
        m.velocity = [float(x) for x in (qd if qd is not None else np.zeros(6))]
        self.js_pub.publish(m)

    # -- callbacks ----------------------------------------------------------
    def on_external_js(self, msg):
        if self.plan is None and set(self.names) <= set(msg.name):
            idx = {n: i for i, n in enumerate(msg.name)}
            self.q = np.array([msg.position[idx[n]] for n in self.names])

    def on_pose(self, msg):
        if self.plan is not None:
            self.status("FAIL: trayectoria en curso; se ignora la nueva pose", error=True)
            return
        if msg.header.frame_id not in ('', self.base_frame):
            self.status(f"FAIL: frame '{msg.header.frame_id}' != '{self.base_frame}' (no se hace transformacion TF)", error=True)
            return
        o, pos = msg.pose.orientation, msg.pose.position
        try:
            T = pl.pose_to_T([pos.x, pos.y, pos.z], [o.x, o.y, o.z, o.w])
        except ValueError as e:
            self.status(f"FAIL: pose invalida ({e})", error=True)
            return
        res = pl.plan_to_pose(T, self.q, self.vmax, self.amax, self.lo, self.hi, dt=self.traj_dt)
        if not res['ok']:
            self.status(f"FAIL: {res['reason']}", error=True)
            return
        for w in res['warnings']:
            rospy.logwarn(w)
        self.publish_trajectory(res)
        self.t_start = rospy.Time.now()      # primero el reloj y luego el plan: el timer corre en otro hilo
        self.plan = res
        self.status(f"OK: rama {res['branch']} ({res['n_valid']}/{res['n_solutions']} soluciones validas), "
                    f"T={res['lspb'].T:.3f} s, error FK pos={res['fk_err_pos']:.1e} m ang={res['fk_err_ang']:.1e} rad")

    def publish_trajectory(self, res):
        jt = JointTrajectory()
        jt.header.stamp = rospy.Time.now()
        jt.joint_names = list(self.names)
        for t, q, qd, qdd in zip(res['t'], res['Q'], res['Qd'], res['Qdd']):
            pt = JointTrajectoryPoint()
            pt.positions, pt.velocities, pt.accelerations = list(q), list(qd), list(qdd)
            pt.time_from_start = rospy.Duration(float(t))
            jt.points.append(pt)
        self.traj_pub.publish(jt)

    def on_timer(self, _event):
        plan = self.plan                       # copia local: on_pose corre en otro hilo
        if plan is None:
            if self.publish_js:
                self.publish_state(self.q)
            return
        t = (rospy.Time.now() - self.t_start).to_sec()
        q, qd, _ = plan['lspb'].at(t)
        if self.publish_js:
            self.publish_state(q, qd)
        if t >= plan['lspb'].T:
            self.q = np.array(plan['q_goal'])
            self.plan = None
            self.status("DONE: movimiento terminado")
        else:
            self.q = q


if __name__ == '__main__':
    rospy.init_node('ik_lspb_node')
    IKLSPBNode()
    rospy.spin()
