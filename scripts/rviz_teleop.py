#!/usr/bin/env python3
"""
rviz_teleop.py - Control visual en RViz para el ABB IRB 120 real.

Combina:
  1) InteractiveMarker 3D en RViz (arrastrar flechas X/Y/Z y aros de rotacion sobre el brazo)
     usando cinematica inversa analitica exacta de abb_irb120_support (0.29/0.27/0.07/0.302/0.072).
  2) Brazo fantasma semitransparente en /preview/joint_states para previsualizar antes de mover.
  3) Ventana de control con 6 sliders (J1..J6 en grados) y botones:
       - Traer desde Robot Real
       - Posicion CERO
       - ENVIAR AL ROBOT REAL (publica a /irb120/target_joints_deg -> tcp_mover_node)
"""
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hito2_seguridad import (
    SPEED_TCP_RANGE,
    SPEED_ORI_RANGE,
    AMAX_RANGE_DEG,
    evaluar_hito2,
    formatear_reporte_hito2,
)

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# Limites seguros con 2 deg de margen (coinciden con ROS_TcpMover.mod)
LIMITS_DEG = [
    (-163.0, 163.0),
    (-108.0, 108.0),
    (-108.0, 68.0),
    (-158.0, 158.0),
    (-118.0, 118.0),
    (-398.0, 398.0),
]


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def quat_to_rot(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def fk_flange(q_rad):
    def tf(xyz, rot):
        M = np.eye(4)
        M[:3, :3] = rot
        M[:3, 3] = xyz
        return M

    T = tf([0.0, 0.0, 0.0], _rz(q_rad[0]))
    T = T @ tf([0.0, 0.0, 0.29], _ry(q_rad[1]))
    T = T @ tf([0.0, 0.0, 0.27], _ry(q_rad[2]))
    T = T @ tf([0.0, 0.0, 0.07], _rx(q_rad[3]))
    T = T @ tf([0.302, 0.0, 0.0], _ry(q_rad[4]))
    T = T @ tf([0.072, 0.0, 0.0], _rx(q_rad[5]))
    return T


def ik_flange(T, q_ref_rad):
    p = T[:3, 3]
    R = T[:3, :3]
    pwc = p - 0.072 * R[:, 0]
    L2 = 0.27
    L3 = math.hypot(0.302, 0.07)
    alpha0 = math.atan2(0.302, 0.07)
    sols = []

    r_xy = math.hypot(pwc[0], pwc[1])
    if r_xy < 1e-6:
        q1_list = [q_ref_rad[0]]
    else:
        q1_list = [math.atan2(pwc[1], pwc[0]), _wrap(math.atan2(pwc[1], pwc[0]) + math.pi)]

    limits_rad = [(math.radians(lo), math.radians(hi)) for lo, hi in LIMITS_DEG]

    for q1 in q1_list:
        r = math.cos(q1) * pwc[0] + math.sin(q1) * pwc[1]
        z = pwc[2] - 0.29
        cphi = (r * r + z * z - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
        if abs(cphi) > 1.0 + 1e-5:
            continue
        cphi = max(-1.0, min(1.0, cphi))
        sphi = math.sqrt(max(0.0, 1.0 - cphi * cphi))
        for sign_elbow in (1.0, -1.0):
            phi = math.atan2(sign_elbow * sphi, cphi)
            q3 = _wrap(phi - alpha0)
            k1 = L2 + L3 * math.cos(phi)
            k2 = L3 * math.sin(phi)
            q2 = _wrap(math.atan2(r, z) - math.atan2(k2, k1))
            R3 = _rz(q1) @ _ry(q2 + q3)
            Rw = R3.T @ R
            c5 = max(-1.0, min(1.0, Rw[0, 0]))
            s5_mag = math.hypot(Rw[1, 0], Rw[2, 0])
            if s5_mag < 1e-5:
                q5 = 0.0 if c5 > 0 else math.pi
                q4 = q_ref_rad[3]
                if q5 == 0.0:
                    q6 = _wrap(math.atan2(Rw[2, 1], Rw[1, 1]) - q4)
                else:
                    q6 = _wrap(math.atan2(-Rw[2, 1], -Rw[1, 1]) + q4)
                wrist = [(q4, q5, q6)]
            else:
                wrist = []
                for s5_sign in (1.0, -1.0):
                    s5 = s5_sign * s5_mag
                    q5 = math.atan2(s5, c5)
                    q4 = math.atan2(Rw[1, 0] / s5, -Rw[2, 0] / s5)
                    q6 = math.atan2(Rw[0, 1] / s5, Rw[0, 2] / s5)
                    wrist.append((q4, q5, q6))
            for q4, q5, q6 in wrist:
                # Permitir giros continuos de J6 dentro de [-398, 398]
                q6_near = q6 + round((q_ref_rad[5] - q6) / (2.0 * math.pi)) * (2.0 * math.pi)
                for q6_cand in (q6, q6_near):
                    cand = [q1, q2, q3, q4, q5, q6_cand]
                    if all(lo <= v <= hi for v, (lo, hi) in zip(cand, limits_rad)):
                        sols.append(cand)

    if not sols:
        return None
    return min(sols, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, q_ref_rad)))


def make_pose_msg(T):
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])
    qx, qy, qz, qw = rot_to_quat(T[:3, :3])
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)
    return pose


class RVizTeleopController:
    def __init__(self, use_gui=True):
        self.lock = threading.Lock()
        self.real_q_deg = [0.0] * 6
        self.has_real = False
        self.preview_q_deg = [0.0] * 6
        self.ik_ok = True
        self.real_pose_mm = None
        self.robot_status_txt = 'Esperando datos del ABB...'
        self.send_status_txt = 'Listo para enviar (usa ENVIAR AL ROBOT REAL)'
        self.gui_dirty = True
        self.pending_confirm = False
        self.use_gui = use_gui
        self.v_tcp_mms = 30.0
        self.v_ori_degs = 15.0
        self.amax_degs2 = 30.0

        self.pub_preview = rospy.Publisher('/preview/joint_states', JointState, queue_size=10)
        self.pub_speed = rospy.Publisher('/irb120/set_speed', Float64MultiArray, queue_size=5)
        self.pub_target = rospy.Publisher('/irb120/target_joints_deg', Float64MultiArray, queue_size=5)

        self.sub_real = rospy.Subscriber('/irb120/raw_joints_deg', Float64MultiArray, self._on_real_joints, queue_size=10)
        self.sub_rpose = rospy.Subscriber('/irb120/real_pose', PoseStamped, self._on_real_pose, queue_size=10)
        self.sub_rstat = rospy.Subscriber('/irb120/robot_status', String, self._on_robot_status, queue_size=10)
        self.sub_sstat = rospy.Subscriber('/irb120/send_status', String, self._on_send_status, queue_size=10)

        self.server = InteractiveMarkerServer('abb_rviz_marker')
        self.menu = MenuHandler()
        self.menu.insert('Enviar movimiento al robot real', callback=self._on_menu_send)
        self.menu.insert('Sincronizar marcador con robot real', callback=self._on_menu_sync)
        self.menu.insert('Ir a posicion CERO (0,0,0,0,0,0)', callback=self._on_menu_zero)
        self._build_marker()

        self.timer = rospy.Timer(rospy.Duration(0.10), self._on_timer)
        rospy.loginfo('rviz_teleop listo! Arrastra el marcador 3D en RViz o usa los sliders.')

        if use_gui:
            self._run_tk_gui()
        else:
            rospy.spin()

    def _on_real_joints(self, msg):
        if len(msg.data) != 6:
            return
        with self.lock:
            first = not self.has_real
            self.real_q_deg = [float(x) for x in msg.data]
            self.has_real = True
            if first:
                self.preview_q_deg = list(self.real_q_deg)
                self._update_marker_pose_locked()
                self.gui_dirty = True

    def _on_real_pose(self, msg):
        p = msg.pose.position
        with self.lock:
            self.real_pose_mm = (float(p.x) * 1000.0, float(p.y) * 1000.0, float(p.z) * 1000.0)

    def _on_robot_status(self, msg):
        with self.lock:
            self.robot_status_txt = msg.data

    def _on_send_status(self, msg):
        with self.lock:
            self.send_status_txt = msg.data

    def _build_marker(self):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = 'base_link'
        int_marker.name = 'irb120_target'
        int_marker.description = 'Arrastra para mover objetivo (clic derecho: enviar)'
        int_marker.scale = 0.22
        int_marker.pose = make_pose_msg(fk_flange([0.0] * 6))

        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = 0.045
        sphere.scale.y = 0.045
        sphere.scale.z = 0.045
        sphere.color.r = 0.1
        sphere.color.g = 0.65
        sphere.color.b = 1.0
        sphere.color.a = 0.85

        menu_ctrl = InteractiveMarkerControl()
        menu_ctrl.interaction_mode = InteractiveMarkerControl.MENU
        menu_ctrl.always_visible = True
        menu_ctrl.markers.append(sphere)
        int_marker.controls.append(menu_ctrl)

        axes = [
            ('x', 1, 0, 0, 1),
            ('y', 0, 0, 1, 1),
            ('z', 0, 1, 0, 1),
        ]
        for name, qx, qy, qz, qw in axes:
            c_rot = InteractiveMarkerControl()
            c_rot.name = f'rotate_{name}'
            c_rot.orientation.x = qx
            c_rot.orientation.y = qy
            c_rot.orientation.z = qz
            c_rot.orientation.w = qw
            c_rot.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(c_rot)

            c_mov = InteractiveMarkerControl()
            c_mov.name = f'move_{name}'
            c_mov.orientation.x = qx
            c_mov.orientation.y = qy
            c_mov.orientation.z = qz
            c_mov.orientation.w = qw
            c_mov.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(c_mov)

        self.server.insert(int_marker, self._on_marker_feedback)
        self.menu.apply(self.server, 'irb120_target')
        self.server.applyChanges()

    def _update_marker_pose_locked(self):
        T = fk_flange([math.radians(v) for v in self.preview_q_deg])
        self.server.setPose('irb120_target', make_pose_msg(T))
        self.server.applyChanges()

    def _on_marker_feedback(self, feedback):
        if feedback.event_type not in (
            InteractiveMarkerFeedback.POSE_UPDATE,
            InteractiveMarkerFeedback.MOUSE_UP,
        ):
            return
        p = feedback.pose.position
        q = feedback.pose.orientation
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
        T[:3, 3] = [p.x, p.y, p.z]

        with self.lock:
            q_ref = [math.radians(v) for v in self.preview_q_deg]
            sol = ik_flange(T, q_ref)
            if sol is not None and T[2, 3] >= 0.0:
                self.preview_q_deg = [math.degrees(v) for v in sol]
                self.ik_ok = True
                self.gui_dirty = True
                if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                    self._update_marker_pose_locked()
            else:
                self.ik_ok = False
                if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                    self._update_marker_pose_locked()

    def _on_menu_send(self, _feedback):
        if self.use_gui:
            with self.lock:
                self.pending_confirm = True
        else:
            self.send_preview_to_real()

    def _on_menu_sync(self, _feedback):
        self.sync_from_real()

    def _on_menu_zero(self, _feedback):
        self.set_zero()

    def sync_from_real(self):
        with self.lock:
            self.preview_q_deg = list(self.real_q_deg)
            self.ik_ok = True
            self._update_marker_pose_locked()
            self.gui_dirty = True

    def set_zero(self):
        with self.lock:
            self.preview_q_deg = [0.0] * 6
            self.ik_ok = True
            self._update_marker_pose_locked()
            self.gui_dirty = True

    def set_preview_from_sliders(self, vals_deg):
        with self.lock:
            self.preview_q_deg = list(vals_deg)
            self.ik_ok = True
            self._update_marker_pose_locked()

    def set_speed_params(self, v_tcp_mms, v_ori_degs, amax_degs2):
        with self.lock:
            self.v_tcp_mms = float(v_tcp_mms)
            self.v_ori_degs = float(v_ori_degs)
            self.amax_degs2 = float(amax_degs2)

    def evaluate_current_preview(self):
        with self.lock:
            target = list(self.preview_q_deg)
            cur = list(self.real_q_deg) if self.has_real else None
            rpose = self.real_pose_mm
            v_tcp = self.v_tcp_mms
            v_ori = self.v_ori_degs
            amax = self.amax_degs2
        ev = evaluar_hito2(
            target,
            q_current_deg=cur,
            v_tcp_mms=v_tcp,
            v_ori_degs=v_ori,
            amax_degs2=amax,
        )
        return target, ev, rpose

    def send_preview_to_real(self):
        target, ev, _ = self.evaluate_current_preview()
        if not ev['aprobado']:
            razones = [c['nombre'] + ': ' + c['detalle'] for c in ev['checks'] if not c['ok']]
            with self.lock:
                self.send_status_txt = 'BLOQUEADO POR SEGURIDAD -> ' + ' | '.join(razones)
            return False
        spd = ev['speed_cfg']
        with self.lock:
            self.send_status_txt = (
                f'Autorizado! Enviando vel=({spd["v_tcp_mms"]:.0f} mm/s, {spd["v_ori_degs"]:.0f} °/s) '
                f'a {[round(x, 1) for x in target]}...'
            )
        self.pub_speed.publish(Float64MultiArray(data=[spd['v_tcp_mms'], spd['v_ori_degs']]))
        rospy.sleep(0.05)
        self.pub_target.publish(Float64MultiArray(data=target))
        return True

    def _on_timer(self, _event):
        with self.lock:
            q_rad = [math.radians(v) for v in self.preview_q_deg]
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = JOINT_NAMES
        js.position = q_rad
        self.pub_preview.publish(js)

    def _run_tk_gui(self):
        root = tk.Tk()
        root.title('Control RViz + Ecuaciones Hito 1/Hito 2 y Velocidades - ABB IRB 120')
        root.geometry('910x840+995+20')

        main_pane = ttk.Panedwindow(root, orient='horizontal')
        main_pane.pack(fill='both', expand=True, padx=6, pady=6)

        left_col = ttk.Frame(main_pane, width=385)
        right_col = ttk.Frame(main_pane, width=515)
        main_pane.add(left_col, weight=2)
        main_pane.add(right_col, weight=3)

        lbl_info = tk.Label(
            left_col,
            text='1) Arrastra el marcador 3D en RViz (Interact) o mueve sliders\n'
                 '2) Ajusta velocidad TCP/Ori/LSPB y mira ecuaciones Hito 2\n'
                 '3) Pulsa PEDIR PERMISO Y MOVER -> gatillo + Play',
            justify='left',
            fg='#1a3a5c',
            padx=6,
            pady=4,
        )
        lbl_info.pack(fill='x')

        slider_vars = []
        suppress_cb = {'flag': False}

        def on_slider(_val=None):
            if suppress_cb['flag']:
                return
            vals = [v.get() for v in slider_vars]
            self.set_preview_from_sliders(vals)

        frame_sliders = ttk.Frame(left_col, padding=(6, 2))
        frame_sliders.pack(fill='x')

        for i, (lo, hi) in enumerate(LIMITS_DEG):
            row = ttk.Frame(frame_sliders)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=f'J{i + 1} (°)', width=7).pack(side='left')
            var = tk.DoubleVar(value=0.0)
            slider_vars.append(var)
            sc = tk.Scale(
                row,
                from_=lo,
                to=hi,
                resolution=0.5,
                orient='horizontal',
                variable=var,
                command=on_slider,
                length=185,
            )
            sc.pack(side='left', fill='x', expand=True)
            lbl_val = ttk.Label(row, text='0.0°', width=8)
            lbl_val.pack(side='right')
            var._lbl = lbl_val

        # Sliders de velocidad del brazo real y perfil LSPB (Hito 2)
        frame_speed = ttk.LabelFrame(left_col, text='Velocidades del Brazo Real (Hito 2)', padding=(6, 4))
        frame_speed.pack(fill='x', padx=6, pady=4)

        v_tcp_var = tk.DoubleVar(value=self.v_tcp_mms)
        v_ori_var = tk.DoubleVar(value=self.v_ori_degs)
        amax_var = tk.DoubleVar(value=self.amax_degs2)

        def on_speed_change(_val=None):
            self.set_speed_params(v_tcp_var.get(), v_ori_var.get(), amax_var.get())

        speed_specs = [
            ('Vel TCP', v_tcp_var, SPEED_TCP_RANGE[0], SPEED_TCP_RANGE[1], 'mm/s'),
            ('Vel Ori', v_ori_var, SPEED_ORI_RANGE[0], SPEED_ORI_RANGE[1], '°/s'),
            ('Ac LSPB', amax_var, AMAX_RANGE_DEG[0], AMAX_RANGE_DEG[1], '°/s²'),
        ]
        for label_txt, var, lo, hi, unit in speed_specs:
            row = ttk.Frame(frame_speed)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=label_txt, width=8).pack(side='left')
            sc = tk.Scale(
                row,
                from_=lo,
                to=hi,
                resolution=5.0,
                orient='horizontal',
                variable=var,
                command=on_speed_change,
                length=165,
            )
            sc.pack(side='left', fill='x', expand=True)
            lbl_u = ttk.Label(row, text=f'{var.get():.0f} {unit}', width=10)
            lbl_u.pack(side='right')
            var._lbl = lbl_u
            var._unit = unit

        preset_row = ttk.Frame(frame_speed)
        preset_row.pack(fill='x', pady=(3, 0))

        def apply_preset(vt, vo, am):
            v_tcp_var.set(vt)
            v_ori_var.set(vo)
            amax_var.set(am)
            on_speed_change()

        ttk.Button(preset_row, text='Lento (20/10)', command=lambda: apply_preset(20, 10, 20)).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(preset_row, text='Normal (30/15)', command=lambda: apply_preset(30, 15, 30)).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(preset_row, text='Rapido (120/50)', command=lambda: apply_preset(120, 50, 90)).pack(side='left', expand=True, fill='x', padx=2)

        lbl_xyz = tk.Label(left_col, text='Objetivo XYZ: ---', font=('Monospace', 9), anchor='w', padx=8)
        lbl_xyz.pack(fill='x', pady=(3, 0))

        lbl_real = tk.Label(left_col, text='Robot real: esperando RWS...', font=('Monospace', 9), anchor='w', padx=8, fg='#333')
        lbl_real.pack(fill='x')

        lbl_seg = tk.Label(left_col, text='SEGURIDAD H1+H2: ---', font=('Sans', 9, 'bold'), anchor='w', padx=8, pady=3, wraplength=365, justify='left')
        lbl_seg.pack(fill='x')

        lbl_stat = tk.Label(left_col, text='', font=('Sans', 9, 'bold'), anchor='w', padx=8, pady=2, fg='#0055aa', wraplength=365, justify='left')
        lbl_stat.pack(fill='x')

        btns = ttk.Frame(left_col, padding=(6, 3))
        btns.pack(fill='x')
        ttk.Button(btns, text='Traer de Robot Real', command=self.sync_from_real).pack(side='left', expand=True, fill='x', padx=3)
        ttk.Button(btns, text='Posicion CERO (0°)', command=self.set_zero).pack(side='left', expand=True, fill='x', padx=3)

        report_mode = tk.StringVar(value='hito2')
        show_step_mats = tk.BooleanVar(value=False)

        def open_confirm_dialog():
            target, ev, rpose = self.evaluate_current_preview()
            dlg_mode = report_mode.get()
            report = formatear_reporte_hito2(ev, modo=dlg_mode, detallado=True, robtarget_real_mm=rpose)

            dlg = tk.Toplevel(root)
            dlg.title('Permiso de Movimiento - Hito 1 y Hito 2 ABB IRB 120')
            dlg.geometry('800x600')
            dlg.transient(root)
            dlg.grab_set()

            txt_dlg = scrolledtext.ScrolledText(dlg, font=('Monospace', 9), wrap='none')
            txt_dlg.pack(fill='both', expand=True, padx=10, pady=8)
            txt_dlg.insert('1.0', report)
            txt_dlg.configure(state='disabled')

            bar = ttk.Frame(dlg, padding=8)
            bar.pack(fill='x')

            pt = ev['hito1']['fk_target']['pos_mm']
            spd = ev['speed_cfg']
            T_tot = ev['lspb']['T']
            if ev['aprobado']:
                lbl_ask = tk.Label(
                    bar,
                    text=f'¿Autoriza mover a ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}) mm a {spd["v_tcp_mms"]:.0f} mm/s (T={T_tot:.2f}s)?',
                    font=('Sans', 10, 'bold'),
                    fg='#1e6f34',
                )
                lbl_ask.pack(side='left', padx=6)

                def do_confirm():
                    dlg.destroy()
                    self.send_preview_to_real()

                tk.Button(
                    bar,
                    text='AUTORIZAR Y MOVER',
                    command=do_confirm,
                    bg='#1e8e3e',
                    fg='white',
                    font=('Sans', 10, 'bold'),
                    padx=12,
                    pady=6,
                ).pack(side='right', padx=4)
            else:
                tk.Label(
                    bar,
                    text='MOVIMIENTO BLOQUEADO POR SEGURIDAD',
                    font=('Sans', 10, 'bold'),
                    fg='#b31b1b',
                ).pack(side='left', padx=6)

            ttk.Button(bar, text='Cancelar', command=dlg.destroy).pack(side='right', padx=4)

        btn_send = tk.Button(
            left_col,
            text='PEDIR PERMISO Y MOVER ROBOT REAL',
            command=open_confirm_dialog,
            bg='#1e8e3e',
            fg='white',
            font=('Sans', 11, 'bold'),
            pady=9,
        )
        btn_send.pack(fill='x', padx=8, pady=(5, 6))

        # Panel derecho: Ecuaciones y calculos matematicos en vivo
        hdr_right = ttk.Frame(right_col)
        hdr_right.pack(fill='x', pady=(0, 2))
        ttk.Label(hdr_right, text='Calculos en vivo (desde RViz)', font=('Sans', 10, 'bold')).pack(side='left', padx=2)
        ttk.Checkbutton(hdr_right, text='Ver matrices/detalle', variable=show_step_mats).pack(side='right', padx=2)

        hdr_modes = ttk.Frame(right_col)
        hdr_modes.pack(fill='x', pady=(0, 4))
        ttk.Radiobutton(hdr_modes, text='Hito 2 (IK+LSPB+Jac+Din)', variable=report_mode, value='hito2').pack(side='left', padx=4)
        ttk.Radiobutton(hdr_modes, text='Todo (H1+H2)', variable=report_mode, value='todo').pack(side='left', padx=4)
        ttk.Radiobutton(hdr_modes, text='Solo Hito 1 (FK)', variable=report_mode, value='hito1').pack(side='left', padx=4)

        txt_live = scrolledtext.ScrolledText(right_col, font=('Monospace', 8), wrap='none')
        txt_live.pack(fill='both', expand=True)
        txt_live.configure(state='disabled')

        last_eval_cache = {'key': None, 'ev': None}
        last_report_key = {'key': None}
        last_err_shown = {'txt': None}

        def refresh_ui():
            if rospy.is_shutdown():
                root.destroy()
                return
            with self.lock:
                pq = list(self.preview_q_deg)
                rq = list(self.real_q_deg)
                has_r = self.has_real
                rpose = self.real_pose_mm
                ik_ok = self.ik_ok
                rstat = self.robot_status_txt
                sstat = self.send_status_txt
                dirty = self.gui_dirty
                self.gui_dirty = False
                need_dlg = self.pending_confirm
                self.pending_confirm = False
                v_tcp = self.v_tcp_mms
                v_ori = self.v_ori_degs
                amax = self.amax_degs2

            if dirty:
                suppress_cb['flag'] = True
                for v, val in zip(slider_vars, pq):
                    v.set(round(val, 2))
                suppress_cb['flag'] = False

            for v, val in zip(slider_vars, pq):
                v._lbl.configure(text=f'{val:6.1f}°')
            for sv in (v_tcp_var, v_ori_var, amax_var):
                sv._lbl.configure(text=f'{sv.get():.0f} {sv._unit}')

            ekey = (
                tuple(round(x, 2) for x in pq),
                tuple(round(x, 1) for x in rq) if has_r else None,
                round(v_tcp, 1),
                round(v_ori, 1),
                round(amax, 1),
            )
            if ekey != last_eval_cache['key'] or last_eval_cache['ev'] is None:
                last_eval_cache['ev'] = evaluar_hito2(
                    pq,
                    q_current_deg=rq if has_r else None,
                    v_tcp_mms=v_tcp,
                    v_ori_degs=v_ori,
                    amax_degs2=amax,
                )
                last_eval_cache['key'] = ekey
            ev_now = last_eval_cache['ev']

            pt = ev_now['hito1']['fk_target']['pos_mm']
            reach_txt = 'OK' if ik_ok else 'FUERA DE ALCANCE'
            lbl_xyz.configure(
                text=f'Objetivo XYZ: X={pt[0]:5.1f}  Y={pt[1]:5.1f}  Z={pt[2]:5.1f} mm  [{reach_txt}]'
            )
            if has_r:
                lbl_real.configure(text='Robot real (°): ' + ' '.join(f'{x:5.1f}' for x in rq))

            T_tot = ev_now['lspb']['T']
            tau_pk = max(abs(x) for x in ev_now['dyn']['tau_peak'])
            if ev_now['aprobado']:
                lbl_seg.configure(
                    text=(
                        f'SEGURIDAD H1+H2: APROBADO | T={T_tot:.2f}s ({ev_now["lspb"]["tipo"]}) '
                        f'| v={v_tcp:.0f} mm/s | |tau|max={tau_pk:.1f} Nm'
                    ),
                    fg='#1e7e34',
                )
                btn_send.configure(state='normal', bg='#1e8e3e')
            else:
                bad = [c['nombre'] for c in ev_now['checks'] if not c['ok']]
                lbl_seg.configure(text='SEGURIDAD H1+H2: BLOQUEADO -> ' + ', '.join(bad), fg='#c62828')
                btn_send.configure(state='disabled', bg='#9e9e9e')

            err_stat = ('ERROR' in sstat) or ('BLOQUEADO' in sstat)
            lbl_stat.configure(
                text=f'ABB: {rstat}\nEnvio: {sstat}',
                fg='#c62828' if err_stat else '#0055aa',
            )

            step_flag = bool(show_step_mats.get())
            mode_val = report_mode.get()
            rkey = (ekey, mode_val, step_flag)
            if rkey != last_report_key['key']:
                last_report_key['key'] = rkey
                y_scroll = txt_live.yview()
                rep = formatear_reporte_hito2(
                    ev_now,
                    modo=mode_val,
                    detallado=step_flag,
                    robtarget_real_mm=rpose,
                )
                txt_live.configure(state='normal')
                txt_live.delete('1.0', 'end')
                txt_live.insert('1.0', rep)
                txt_live.yview_moveto(y_scroll[0])
                txt_live.configure(state='disabled')

            if need_dlg:
                open_confirm_dialog()
            if sstat.startswith('ERROR:') and sstat != last_err_shown['txt']:
                last_err_shown['txt'] = sstat
                root.after(10, lambda msg_err=sstat: messagebox.showerror('Error al mover robot ABB', msg_err))
            elif not sstat.startswith('ERROR:'):
                last_err_shown['txt'] = None
            root.after(150, refresh_ui)

        root.after(150, refresh_ui)
        root.protocol('WM_DELETE_WINDOW', lambda: (rospy.signal_shutdown('GUI cerrada'), root.destroy()))
        root.mainloop()


if __name__ == '__main__':
    rospy.init_node('rviz_teleop')
    use_gui = bool(rospy.get_param('~use_gui', True))
    RVizTeleopController(use_gui=use_gui)
