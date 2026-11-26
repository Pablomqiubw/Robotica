#!/usr/bin/env python3
"""
tcp_mover.py - Envia objetivos articulares (J1..J6 en grados) y velocidad segura
al ABB IRB 120 por Socket TCP usando el modulo ROS_TcpMover (puerto 5005),
SIN necesidad de Grant / RMMP / mastership.

Uso directo (CLI):
  rosrun irb120_description tcp_mover.py --status
  rosrun irb120_description tcp_mover.py --wait --speed 60 25
  rosrun irb120_description tcp_mover.py --wait --speed 60 25 --delta 0 0 0 0 0 5
  rosrun irb120_description tcp_mover.py --wait --abs 5 0 0 0 0 0

Uso como nodo ROS (escucha /irb120/target_joints_deg y /irb120/set_speed):
  rosrun irb120_description tcp_mover.py
"""
import argparse
import socket
import sys
import time

import requests
from requests.auth import HTTPDigestAuth
import rospy
from std_msgs.msg import Float64MultiArray, String


LIMITS_DEG = [
    (-163.0, 163.0),
    (-108.0, 108.0),
    (-108.0, 68.0),
    (-158.0, 158.0),
    (-118.0, 118.0),
    (-398.0, 398.0),
]

SPEED_TCP_RANGE = (5.0, 150.0)   # mm/s en Manual Reducido (< 250 mm/s ABB)
SPEED_ORI_RANGE = (5.0, 60.0)    # deg/s orientacion segura

ERR_DESC = {
    '-1': 'formato o rango invalido',
    '-2': 'fuera de limites articulares',
    '-3': 'paso mayor al permitido por el modulo RAPID (hasta 20 grados)',
    '-4': 'modo no permitido (requiere Manual Reducido)',
    '-5': 'altura Z menor a 0 mm',
    '-6': 'velocidad fuera de rango seguro (v_tcp 5..150 mm/s, v_ori 5..60 deg/s)',
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class ABBTcpMoverClient:
    def __init__(self, robot_ip='192.168.125.1', tcp_port=5005, rws_port=80, user='Default User', password='robotics'):
        self.ip = robot_ip
        self.tcp_port = int(tcp_port)
        self.base = f'http://{robot_ip}:{rws_port}'
        self.s = requests.Session()
        self.s.auth = HTTPDigestAuth(user, password)

    @staticmethod
    def validate_speed(v_tcp_mms, v_ori_degs):
        v_tcp = float(v_tcp_mms)
        v_ori = float(v_ori_degs)
        if not (SPEED_TCP_RANGE[0] <= v_tcp <= SPEED_TCP_RANGE[1]):
            raise ValueError(
                f'v_tcp={v_tcp:.1f} mm/s fuera de rango seguro [{SPEED_TCP_RANGE[0]}, {SPEED_TCP_RANGE[1]}] mm/s'
            )
        if not (SPEED_ORI_RANGE[0] <= v_ori <= SPEED_ORI_RANGE[1]):
            raise ValueError(
                f'v_ori={v_ori:.1f} deg/s fuera de rango seguro [{SPEED_ORI_RANGE[0]}, {SPEED_ORI_RANGE[1]}] deg/s'
            )
        return v_tcp, v_ori

    def _get_json(self, path):
        r = self.s.get(self.base + path, timeout=2.0, headers={'Accept': 'application/json'})
        r.raise_for_status()
        return r.json()

    def get_joints_deg(self):
        d = self._get_json('/rw/motionsystem/mechunits/ROB_1/jointtarget?json=1')
        st = d['_embedded']['_state'][0]
        return [float(st[f'rax_{i}']) for i in range(1, 7)]

    def get_status(self):
        ctrlstate = self._get_json('/rw/panel/ctrlstate?json=1')['_embedded']['_state'][0]['ctrlstate']
        opmode = self._get_json('/rw/panel/opmode?json=1')['_embedded']['_state'][0]['opmode']
        execstate = self._get_json('/rw/rapid/execution?json=1')['_embedded']['_state'][0]['ctrlexecstate']
        rob_sync = 'unknown'
        try:
            rob_sync = self._get_json('/rw/motionsystem/mechunits/ROB_1?json=1')['_embedded']['_state'][0].get('status', 'unknown')
        except Exception:
            pass
        pp_mod, pp_rout = None, None
        try:
            pcp = self._get_json('/rw/rapid/tasks/T_ROB1/pcp?json=1')['_embedded']['_state'][0]
            pp_mod = pcp.get('modulemame') or pcp.get('modulename')
            pp_rout = pcp.get('routinename')
        except Exception:
            pass
        tcp_ok = self.ping_tcp()
        try:
            q_deg = self.get_joints_deg()
        except Exception:
            q_deg = None
        return {
            'ctrlstate': ctrlstate,
            'opmode': opmode,
            'execstate': execstate,
            'rob_sync': rob_sync,
            'pp': f'{pp_mod}/{pp_rout}' if pp_mod or pp_rout else None,
            'tcp_5005_open': tcp_ok,
            'q_deg': q_deg,
        }

    def ping_tcp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((self.ip, self.tcp_port))
            sock.sendall(b'PING')
            resp = sock.recv(64).decode('utf-8', errors='ignore').strip()
            return resp == 'PONG'
        except Exception:
            return False
        finally:
            sock.close()

    def check_ready(self):
        st = self.get_status()
        problems = []
        if st.get('q_deg') is None or st.get('rob_sync') == 'NotAMSSynchronized':
            problems.append("ROB_1 no sincronizado (actualiza cuentarrevoluciones en Calibration -> ROB_1 -> Update Revolution Counters)")
        if st['opmode'] != 'MANR':
            problems.append(f"opmode es '{st['opmode']}' (requiere 'MANR')")
        if st.get('pp') and st['pp'] not in ('ROS_TcpMover/ServidorTCP', 'ROS_TcpMover/AtenderCliente'):
            problems.append(f"PP apunta a '{st['pp']}' (cambia PP a ROS_TcpMover -> ServidorTCP)")
        if st['ctrlstate'] != 'motoron':
            problems.append(f"ctrlstate es '{st['ctrlstate']}' (manten apretado el gatillo trasero del FlexPendant)")
        if st['execstate'] != 'running':
            problems.append("RAPID esta 'stopped' (pulsa Play en ServidorTCP)")
        elif not st['tcp_5005_open']:
            problems.append("puerto TCP 5005 aun no responde (toca 'Iniciar' en el FlexPendant o apunta PP a ServidorTCP)")
        return st, problems

    def wait_until_ready(self, wait_sec=120.0):
        t0 = time.time()
        last_msg = ''
        while time.time() - t0 < wait_sec:
            st, problems = self.check_ready()
            if not problems:
                print('ServidorTCP listo en puerto 5005! Enviando movimiento...', flush=True)
                return st
            msg = ' | '.join(problems)
            if msg != last_msg:
                print('Esperando al FlexPendant:', msg, flush=True)
                last_msg = msg
            time.sleep(0.35)
        raise TimeoutError('Tiempo agotado esperando gatillo + Play + Iniciar en ServidorTCP')

    def _send_speed_on_sock(self, sock, v_tcp_mms, v_ori_degs):
        v_tcp, v_ori = self.validate_speed(v_tcp_mms, v_ori_degs)
        payload = f'V:[{v_tcp:.1f},{v_ori:.1f},{v_tcp:.1f},{v_ori:.1f}]'
        sock.sendall(payload.encode('ascii'))
        resp = sock.recv(128).decode('utf-8', errors='ignore').strip()
        if resp == 'OK':
            return 'OK'
        if resp.startswith('ERR:'):
            code = resp.split(':', 1)[1].strip()
            if code == '-1':
                raise RuntimeError(
                    'El controlador rechazo comando V: (ERR:-1). '
                    'Recarga ROS_TcpMover.mod en el FlexPendant (Modules -> ROS_TcpMover -> File -> Load Module) y pulsa Play.'
                )
            raise RuntimeError(f'Robot rechazo velocidad ({resp}): {ERR_DESC.get(code, "error")}')
        raise RuntimeError(f'Respuesta inesperada por TCP al ajustar velocidad: {resp!r}')

    def set_speed(self, v_tcp_mms, v_ori_degs, wait_sec=0.0):
        v_tcp, v_ori = self.validate_speed(v_tcp_mms, v_ori_degs)
        if wait_sec > 0:
            self.wait_until_ready(wait_sec=wait_sec)
        else:
            _, problems = self.check_ready()
            if problems:
                raise RuntimeError('Robot no listo:\n  - ' + '\n  - '.join(problems))

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        try:
            sock.connect((self.ip, self.tcp_port))
            return self._send_speed_on_sock(sock, v_tcp, v_ori)
        finally:
            sock.close()

    def _send_single_step(self, sock, target_deg):
        for i, (val, (lo, hi)) in enumerate(zip(target_deg, LIMITS_DEG), start=1):
            if not (lo <= val <= hi):
                raise ValueError(f'Eje J{i}={val:.2f} fuera de limite seguro [{lo}, {hi}]')
        payload = '[' + ','.join(f'{v:.3f}' for v in target_deg) + ']'
        sock.sendall(payload.encode('ascii'))
        resp = sock.recv(128).decode('utf-8', errors='ignore').strip()
        if resp == 'OK':
            return 'OK'
        if resp.startswith('ERR:'):
            code = resp.split(':', 1)[1].strip()
            raise RuntimeError(f'Robot rechazo orden ({resp}): {ERR_DESC.get(code, "error")}')
        raise RuntimeError(f'Respuesta inesperada por TCP: {resp!r}')

    def move_abs_deg(self, target_deg, max_step_deg=15.0, dry_run=False, wait_sec=0.0, speed_tcp_ori=None):
        v_tcp, v_ori = None, None
        if speed_tcp_ori is not None:
            v_tcp, v_ori = self.validate_speed(speed_tcp_ori[0], speed_tcp_ori[1])

        if dry_run:
            q = self.get_joints_deg()
            max_diff = max(abs(t - c) for t, c in zip(target_deg, q))
            n_steps = 1 if max_diff <= max_step_deg else int(max_diff // max_step_deg) + 1
            return [[c + (k / n_steps) * (t - c) for c, t in zip(q, target_deg)] for k in range(1, n_steps + 1)]

        if wait_sec > 0:
            st = self.wait_until_ready(wait_sec=wait_sec)
        else:
            st, problems = self.check_ready()
            if problems:
                raise RuntimeError('Robot no listo:\n  - ' + '\n  - '.join(problems))

        current = st['q_deg']
        max_diff = max(abs(t - c) for t, c in zip(target_deg, current))
        n_steps = 1 if max_diff <= max_step_deg else int(max_diff // max_step_deg) + 1

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30.0)
        results = []
        try:
            sock.connect((self.ip, self.tcp_port))
            if v_tcp is not None:
                try:
                    self._send_speed_on_sock(sock, v_tcp, v_ori)
                    results.append(f'vel={v_tcp:.1f}mm/s,{v_ori:.1f}deg/s')
                except RuntimeError as exc:
                    if 'ERR:-1' in str(exc) and abs(v_tcp - 30.0) < 1e-6 and abs(v_ori - 15.0) < 1e-6:
                        results.append('vel=default(30/15, recargar ROS_TcpMover.mod)')
                    else:
                        raise
            for k in range(1, n_steps + 1):
                alpha = float(k) / float(n_steps)
                wp = [c + alpha * (t - c) for c, t in zip(current, target_deg)]
                res = self._send_single_step(sock, wp)
                results.append(f'paso {k}/{n_steps}: {res}')
        finally:
            sock.close()
        return results


class TCPSendNode:
    def __init__(self):
        p = rospy.get_param
        self.client = ABBTcpMoverClient(
            robot_ip=str(p('~robot_ip', '192.168.125.1')),
            tcp_port=int(p('~tcp_port', 5005)),
        )
        self.max_step_deg = float(p('~max_step_deg', 15.0))
        self.wait_sec = float(p('~wait_sec', 25.0))
        v_tcp_def = float(p('~default_v_tcp', 30.0))
        v_ori_def = float(p('~default_v_ori', 15.0))
        self.target_speed = ABBTcpMoverClient.validate_speed(v_tcp_def, v_ori_def)

        self.pub_status = rospy.Publisher('/irb120/send_status', String, queue_size=10)
        self.sub_speed = rospy.Subscriber('/irb120/set_speed', Float64MultiArray, self._on_speed, queue_size=1)
        self.sub = rospy.Subscriber('/irb120/target_joints_deg', Float64MultiArray, self._on_target, queue_size=1)
        rospy.loginfo(
            'tcp_mover_node listo (puerto 5005, max_step=%.1f deg, vel=%.1f mm/s, %.1f deg/s). '
            'Escuchando /irb120/set_speed y /irb120/target_joints_deg.',
            self.max_step_deg, self.target_speed[0], self.target_speed[1],
        )

    def _on_speed(self, msg):
        if len(msg.data) != 2:
            txt = f'ERROR velocidad: se esperaban [v_tcp_mms, v_ori_degs], llegaron {len(msg.data)}'
            rospy.logerr(txt)
            self.pub_status.publish(String(data=txt))
            return
        try:
            self.target_speed = ABBTcpMoverClient.validate_speed(msg.data[0], msg.data[1])
            txt = f'Velocidad programada: {self.target_speed[0]:.1f} mm/s, {self.target_speed[1]:.1f} deg/s'
            rospy.loginfo(txt)
            self.pub_status.publish(String(data=txt))
        except Exception as exc:
            txt = f'ERROR velocidad: {exc}'
            rospy.logerr(txt)
            self.pub_status.publish(String(data=txt))

    def _on_target(self, msg):
        if len(msg.data) != 6:
            txt = f'ERROR: se esperaban 6 angulos en grados, llegaron {len(msg.data)}'
            rospy.logerr(txt)
            self.pub_status.publish(String(data=txt))
            return
        try:
            v_tcp, v_ori = self.target_speed
            self.pub_status.publish(
                String(data=f'Esperando gatillo + Play (vel {v_tcp:.0f} mm/s, max {self.wait_sec:.0f}s) y enviando...')
            )
            res = self.client.move_abs_deg(
                list(msg.data),
                max_step_deg=self.max_step_deg,
                wait_sec=self.wait_sec,
                speed_tcp_ori=self.target_speed,
            )
            txt = 'OK: ' + ' | '.join(res)
            rospy.loginfo(txt)
            self.pub_status.publish(String(data=txt))
        except Exception as exc:
            txt = f'ERROR: {exc}'
            rospy.logerr(txt)
            self.pub_status.publish(String(data=txt))


def main():
    try:
        _logf = open('/tmp/tcp_mover_last.log', 'w')
        sys.stdout = _Tee(sys.__stdout__, _logf)
        sys.stderr = _Tee(sys.__stderr__, _logf)
    except Exception:
        pass

    args = rospy.myargv(argv=sys.argv)[1:]
    if not args:
        rospy.init_node('tcp_mover_node')
        TCPSendNode()
        rospy.spin()
        return

    parser = argparse.ArgumentParser(description='Enviar movimiento seguro a ABB IRB 120 por TCP (puerto 5005)')
    parser.add_argument('--ip', default='192.168.125.1')
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--speed', nargs=2, type=float, metavar=('V_TCP_MMS', 'V_ORI_DEGS'),
                        help='Velocidad segura [5..150 mm/s, 5..60 deg/s]')
    parser.add_argument('--abs', nargs=6, type=float, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'))
    parser.add_argument('--delta', nargs=6, type=float, metavar=('D1', 'D2', 'D3', 'D4', 'D5', 'D6'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wait', action='store_true')
    parsed = parser.parse_args(args)

    client = ABBTcpMoverClient(robot_ip=parsed.ip, tcp_port=parsed.port)
    speed_pair = None
    if parsed.speed is not None:
        speed_pair = ABBTcpMoverClient.validate_speed(parsed.speed[0], parsed.speed[1])

    if parsed.status:
        st, problems = client.check_ready()
        print('Estado IRC5/TCP:', st)
        if problems:
            print('Falta:')
            for pr in problems:
                print(' -', pr)
        else:
            print('LISTO POR TCP 5005!')
        return

    if parsed.abs is None and parsed.delta is None:
        if speed_pair is not None:
            res = client.set_speed(speed_pair[0], speed_pair[1], wait_sec=120.0 if parsed.wait else 0.0)
            print('Velocidad actualizada:', speed_pair, res)
        else:
            parser.print_help()
        return

    q_now = client.get_joints_deg()
    if parsed.abs is not None:
        target = list(parsed.abs)
    else:
        target = [c + d for c, d in zip(q_now, parsed.delta)]

    print('Actual (deg):', [round(x, 3) for x in q_now])
    print('Objetivo (deg):', [round(x, 3) for x in target])
    if speed_pair is not None:
        print('Velocidad (mm/s, deg/s):', speed_pair)
    res = client.move_abs_deg(
        target,
        dry_run=parsed.dry_run,
        wait_sec=120.0 if parsed.wait else 0.0,
        speed_tcp_ori=speed_pair,
    )
    print('Resultado:', res)


if __name__ == '__main__':
    main()
