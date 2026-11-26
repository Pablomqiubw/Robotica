#!/usr/bin/env python3
"""
abb_receive_node.py - Nodo ROS (Noetic) para recibir datos en vivo del ABB IRB 120 (IRC5).

Modos (~mode):
  rws        : Lee Robot Web Services HTTP de ABB (RobotWare 6/7) en http://<robot_ip>:<rws_port>
  tcp_client : Se conecta por Socket TCP a <robot_ip>:<tcp_port> y lee 6 valores por linea
  demo       : Genera movimiento suave local para probar RViz/topicos/CSV sin robot conectado

Topicos publicados:
  /joint_states          (sensor_msgs/JointState)       - Para mover el brazo 3D en RViz
  /irb120/raw_joints_deg (std_msgs/Float64MultiArray)   - J1..J6 en grados
  /irb120/real_pose      (geometry_msgs/PoseStamped)    - Pose cartesiana (m) en base_link
  /irb120/robot_status   (std_msgs/String)              - Estado de conexion y controlador
"""
import csv
import math
import os
import re
import socket
import sys
import time

import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from irb120_fk import forward_kinematics
except Exception:
    forward_kinematics = None


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']


def _rot_to_quat(R):
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


def _extract_state_dict(payload):
    if not isinstance(payload, dict):
        return {}
    emb = payload.get('_embedded', {})
    if isinstance(emb, dict):
        state = emb.get('_state') or emb.get('state')
        if isinstance(state, list) and state:
            return state[0]
    state = payload.get('state')
    if isinstance(state, list) and state:
        return state[0]
    return payload


class ABBReceiveNode:
    def __init__(self):
        p = rospy.get_param
        self.mode = str(p('~mode', 'rws')).strip().lower()
        self.robot_ip = str(p('~robot_ip', '192.168.125.1')).strip()
        self.rws_port = int(p('~rws_port', 80))
        self.rws_user = str(p('~rws_user', 'Default User'))
        self.rws_pass = str(p('~rws_pass', 'robotics'))
        self.tcp_port = int(p('~tcp_port', 5000))
        self.tcp_units = str(p('~tcp_units', 'deg')).strip().lower()
        self.rate_hz = float(p('~rate', 15.0))
        self.base_frame = str(p('~base_frame', 'base_link'))
        self.timeout = float(p('~timeout', 1.5))
        self.log_csv = str(p('~log_csv', '')).strip()

        self.pub_js = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.pub_deg = rospy.Publisher('/irb120/raw_joints_deg', Float64MultiArray, queue_size=10)
        self.pub_pose = rospy.Publisher('/irb120/real_pose', PoseStamped, queue_size=10)
        self.pub_status = rospy.Publisher('/irb120/robot_status', String, queue_size=10)

        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(self.rws_user, self.rws_pass)
        self.tcp_sock = None
        self.tcp_buf = ''
        self.t0 = time.time()
        self.cycle = 0
        self.last_ctrl_status = 'unknown'

        self.csv_file = None
        self.csv_writer = None
        if self.log_csv:
            csv_dir = os.path.dirname(os.path.abspath(self.log_csv))
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            new_file = not os.path.exists(self.log_csv)
            self.csv_file = open(self.log_csv, 'a', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            if new_file:
                self.csv_writer.writerow(
                    ['ros_time', 'j1_deg', 'j2_deg', 'j3_deg', 'j4_deg', 'j5_deg', 'j6_deg', 'x_m', 'y_m', 'z_m', 'status']
                )
            rospy.loginfo('Guardando datos recibidos en CSV: %s', self.log_csv)

        rospy.loginfo(
            'ABBReceiveNode iniciado | modo=%s | ip=%s | rate=%.1f Hz',
            self.mode, self.robot_ip, self.rate_hz
        )

    def _rws_get(self, path):
        url = f'http://{self.robot_ip}:{self.rws_port}{path}'
        resp = self.session.get(url, timeout=self.timeout, headers={'Accept': 'application/json'})
        if resp.status_code == 400 and 'jointtarget' in path:
            try:
                mu = self.session.get(
                    f'http://{self.robot_ip}:{self.rws_port}/rw/motionsystem/mechunits/ROB_1?json=1',
                    timeout=self.timeout, headers={'Accept': 'application/json'}
                ).json()
                st = _extract_state_dict(mu).get('status', '')
                if st == 'NotAMSSynchronized':
                    raise RuntimeError('Cuentarrevoluciones no actualizado (NotAMSSynchronized) en FlexPendant')
            except RuntimeError:
                raise
            except Exception:
                pass
        resp.raise_for_status()
        return resp.json()

    def _read_rws(self):
        jt_json = self._rws_get('/rw/motionsystem/mechunits/ROB_1/jointtarget?json=1')
        jt = _extract_state_dict(jt_json)
        q_deg = [float(jt[f'rax_{i}']) for i in range(1, 7)]

        pose_xyz_quat = None
        try:
            rt_json = self._rws_get('/rw/motionsystem/mechunits/ROB_1/robtarget?json=1')
            rt = _extract_state_dict(rt_json)
            if all(k in rt for k in ('x', 'y', 'z', 'q1', 'q2', 'q3', 'q4')):
                pose_xyz_quat = (
                    float(rt['x']) / 1000.0,
                    float(rt['y']) / 1000.0,
                    float(rt['z']) / 1000.0,
                    float(rt['q2']),
                    float(rt['q3']),
                    float(rt['q4']),
                    float(rt['q1']),
                )
        except Exception:
            pose_xyz_quat = None

        if self.cycle % max(1, int(self.rate_hz)) == 0:
            try:
                cs = _extract_state_dict(self._rws_get('/rw/panel/ctrlstate?json=1')).get('ctrlstate', '?')
                om = _extract_state_dict(self._rws_get('/rw/panel/opmode?json=1')).get('opmode', '?')
                self.last_ctrl_status = f'RWS_OK ctrl={cs} opmode={om}'
            except Exception:
                self.last_ctrl_status = 'RWS_OK'

        return q_deg, pose_xyz_quat, self.last_ctrl_status

    def _read_tcp(self):
        if self.tcp_sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.robot_ip, self.tcp_port))
            self.tcp_sock = s
            self.tcp_buf = ''
            rospy.loginfo('Conectado por TCP a %s:%d', self.robot_ip, self.tcp_port)

        chunk = self.tcp_sock.recv(1024).decode('utf-8', errors='ignore')
        if not chunk:
            raise ConnectionError('Socket cerrado por el controlador')
        self.tcp_buf += chunk
        if '\n' not in self.tcp_buf:
            return None

        lines = self.tcp_buf.split('\n')
        self.tcp_buf = lines[-1]
        last_line = ''
        for line in reversed(lines[:-1]):
            if line.strip():
                last_line = line.strip()
                break
        if not last_line:
            return None

        nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', last_line)]
        if len(nums) < 6:
            raise ValueError(f'Linea TCP con menos de 6 valores: {last_line!r}')
        q_vals = nums[:6]
        if self.tcp_units == 'rad':
            q_deg = [math.degrees(v) for v in q_vals]
        else:
            q_deg = q_vals
        return q_deg, None, f'TCP_OK {self.robot_ip}:{self.tcp_port}'

    def _read_demo(self):
        t = time.time() - self.t0
        q_deg = [
            25.0 * math.sin(0.6 * t),
            15.0 * math.sin(0.8 * t),
            -20.0 * math.sin(0.7 * t),
            30.0 * math.sin(0.9 * t),
            20.0 * math.cos(0.7 * t),
            40.0 * math.sin(0.5 * t),
        ]
        return q_deg, None, 'DEMO_LOCAL (sin robot fisico)'

    def _publish(self, q_deg, pose_xyz_quat, status_str):
        now = rospy.Time.now()
        q_rad = [math.radians(v) for v in q_deg]

        js = JointState()
        js.header.stamp = now
        js.name = JOINT_NAMES
        js.position = q_rad
        self.pub_js.publish(js)

        arr = Float64MultiArray()
        arr.data = [float(v) for v in q_deg]
        self.pub_deg.publish(arr)

        if pose_xyz_quat is None and forward_kinematics is not None:
            T = forward_kinematics(np.array(q_rad, dtype=float))
            qx, qy, qz, qw = _rot_to_quat(T[:3, :3])
            pose_xyz_quat = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), qx, qy, qz, qw)

        if pose_xyz_quat is not None:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = self.base_frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pose_xyz_quat[:3]
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = pose_xyz_quat[3:]
            self.pub_pose.publish(ps)
            px, py, pz = pose_xyz_quat[:3]
        else:
            px, py, pz = 0.0, 0.0, 0.0

        self.pub_status.publish(String(data=status_str))

        if self.csv_writer is not None:
            self.csv_writer.writerow(
                [f'{now.to_sec():.4f}'] + [f'{v:.3f}' for v in q_deg] + [f'{px:.4f}', f'{py:.4f}', f'{pz:.4f}', status_str]
            )
            if self.cycle % 20 == 0:
                self.csv_file.flush()

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.cycle += 1
            try:
                if self.mode == 'rws':
                    res = self._read_rws()
                elif self.mode == 'tcp_client':
                    res = self._read_tcp()
                elif self.mode == 'demo':
                    res = self._read_demo()
                else:
                    rospy.logerr_throttle(5.0, 'Modo no reconocido: %s (usa rws, tcp_client o demo)', self.mode)
                    rate.sleep()
                    continue

                if res is not None:
                    self._publish(*res)
            except Exception as exc:
                if self.tcp_sock is not None:
                    try:
                        self.tcp_sock.close()
                    except Exception:
                        pass
                    self.tcp_sock = None
                msg = f'ERROR_{self.mode.upper()}: {exc}'
                self.pub_status.publish(String(data=msg))
                rospy.logwarn_throttle(3.0, 'Esperando datos del ABB (%s): %s', self.robot_ip, exc)
            rate.sleep()

        if self.csv_file is not None:
            self.csv_file.close()


if __name__ == '__main__':
    rospy.init_node('abb_receive_node')
    ABBReceiveNode().spin()
