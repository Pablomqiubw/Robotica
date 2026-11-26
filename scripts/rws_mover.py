#!/usr/bin/env python3
"""
rws_mover.py - Envia objetivos articulares (J1..J6 en grados) al ABB IRB 120 via RWS.
Soporta automaticamente los dos modulos cargados en el controlador:
  1) IMT342_DestinoDirecto (ServidorDirecto) - el puntero actual del robot
  2) LabJointRws           (ServidorPC)

Uso directo (CLI):
  rosrun irb120_description rws_mover.py --status
  rosrun irb120_description rws_mover.py --wait --delta 0 0 0 0 0 5
  rosrun irb120_description rws_mover.py --wait --abs 5 0 0 0 0 0

Uso como nodo ROS (escucha /irb120/target_joints_deg):
  rosrun irb120_description rws_mover.py
"""
import argparse
import re
import sys
import time

import requests
from requests.auth import HTTPDigestAuth
import rospy
from std_msgs.msg import Float64MultiArray, String


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


LIMITS_DEG = [
    (-163.0, 163.0),
    (-108.0, 108.0),
    (-108.0, 68.0),
    (-158.0, 158.0),
    (-118.0, 118.0),
    (-398.0, 398.0),
]

LAB_CODE_DESC = {
    0: 'listo',
    1: 'moviendo',
    2: 'llegada OK',
    -1: 'objetivo vacio o no valido',
    -2: 'fuera de limites articulares',
    -3: 'paso mayor a maxPasoDeg (10 deg)',
    -4: 'modo no permitido (debe estar en Manual Reducido)',
}

DIRECTO_CODE_DESC = {
    0: 'listo',
    1: 'moviendo',
    2: 'llegada OK',
    3: 'revisando / esperando confirmacion en FlexPendant',
    -1: 'cancelado o timeout en FlexPendant',
    -2: 'fuera de limites articulares',
    -3: 'posicion inicial cambio antes de mover',
    -5: 'altura Z menor que alturaMinima (0 mm)',
    -6: 'ejes externos no invalidos (9E9)',
    -7: 'error interno de ejecucion',
}


class ABBRwsMoverClient:
    def __init__(self, robot_ip='192.168.125.1', port=80, user='Default User', password='robotics', timeout=2.5):
        self.base = f'http://{robot_ip}:{port}'
        self.user = user
        self.timeout = timeout
        self.s = requests.Session()
        self.s.auth = HTTPDigestAuth(user, password)

    def _get_json(self, path):
        r = self.s.get(self.base + path, timeout=self.timeout, headers={'Accept': 'application/json'})
        r.raise_for_status()
        return r.json()

    def _get_symbol(self, module, name):
        d = self._get_json(f'/rw/rapid/symbol/data/RAPID/T_ROB1/{module}/{name}?json=1')
        return d['_embedded']['_state'][0]['value']

    def _set_symbol_raw(self, module, name, value):
        url = f'{self.base}/rw/rapid/symbol/data/RAPID/T_ROB1/{module}/{name}?action=set'
        r = self.s.post(url, data={'value': str(value)}, timeout=self.timeout)
        r.raise_for_status()

    def _ensure_rmmp(self, wait_sec=120.0):
        rmmp = self._get_json('/users/rmmp?json=1')['_embedded']['_state'][0]
        if rmmp.get('rmmpheldbyme') == 'true':
            return
        try:
            self.s.post(
                f'{self.base}/users',
                data={'username': self.user, 'application': 'ROS_IRB120', 'location': 'Ubuntu_ROS'},
                timeout=self.timeout,
            )
        except Exception:
            pass

        r = self.s.post(f'{self.base}/users/rmmp', data={'privilege': 'modify'}, timeout=self.timeout)
        if r.status_code not in (200, 202, 204):
            r.raise_for_status()

        print('>>> ATENCION: Mira la pantalla del FlexPendant y toca GRANT / CONCEDER para dar permiso a la PC! <<<', flush=True)
        t0 = time.time()
        while time.time() - t0 < wait_sec:
            st = self._get_json('/users/rmmp?json=1')['_embedded']['_state'][0]
            if st.get('rmmpheldbyme') == 'true':
                print('Permiso de escritura (RMMP) concedido!', flush=True)
                return
            try:
                poll = self._get_json('/users/rmmp/poll?json=1')['_embedded']['_state'][0]
                if str(poll.get('status', '')).upper() == 'DENY':
                    raise RuntimeError('Permiso RMMP denegado en el FlexPendant')
            except RuntimeError:
                raise
            except Exception:
                pass
            time.sleep(0.4)
        raise TimeoutError('Tiempo agotado esperando que toques Grant/Conceder en el FlexPendant')

    def _request_rapid_mastership(self):
        st = self._get_json('/rw/mastership/rapid?json=1')['_embedded']['_state'][0]
        if st.get('mastershipheldbyme') == 'TRUE':
            return False
        r = self.s.post(
            f'{self.base}/rw/mastership/rapid?action=request',
            data={},
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=self.timeout,
        )
        if r.status_code in (200, 204):
            return True
        r_all = self.s.post(
            f'{self.base}/rw/mastership?action=request',
            data={},
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=self.timeout,
        )
        if r_all.status_code in (200, 204):
            return True
        r.raise_for_status()
        return True

    def _release_rapid_mastership(self):
        try:
            self.s.post(
                f'{self.base}/rw/mastership/rapid?action=release',
                data={},
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=self.timeout,
            )
        except Exception:
            pass

    def _write_symbols(self, module, pairs):
        self._ensure_rmmp()
        acquired = False
        try:
            for name, value in pairs:
                try:
                    self._set_symbol_raw(module, name, value)
                except requests.HTTPError as err:
                    if err.response is not None and err.response.status_code == 403 and not acquired:
                        try:
                            acquired = self._request_rapid_mastership()
                        except requests.HTTPError as m_err:
                            body = m_err.response.text if m_err.response is not None else ''
                            raise RuntimeError(
                                f'ABB rechazo acceso (403). Cierra Program Editor (toca X) o abre Production Window en el FlexPendant. Detalle: {body}'
                            )
                        self._set_symbol_raw(module, name, value)
                    else:
                        raise
        finally:
            if acquired:
                self._release_rapid_mastership()

    def get_joints_deg(self):
        d = self._get_json('/rw/motionsystem/mechunits/ROB_1/jointtarget?json=1')
        st = d['_embedded']['_state'][0]
        return [float(st[f'rax_{i}']) for i in range(1, 7)]

    def get_status(self):
        ctrlstate = self._get_json('/rw/panel/ctrlstate?json=1')['_embedded']['_state'][0]['ctrlstate']
        opmode = self._get_json('/rw/panel/opmode?json=1')['_embedded']['_state'][0]['opmode']
        execstate = self._get_json('/rw/rapid/execution?json=1')['_embedded']['_state'][0]['ctrlexecstate']
        pcp_mod = ''
        pcp_rout = ''
        try:
            pcp = self._get_json('/rw/rapid/tasks/T_ROB1/pcp?json=1')['_embedded']['_state'][0]
            pcp_mod = pcp.get('modulemame') or pcp.get('modulename') or ''
            pcp_rout = pcp.get('routinename') or ''
        except Exception:
            pass

        rb_activo = int(float(self._get_symbol('LabJointRws', 'rbActivo')))
        directo_activo = int(float(self._get_symbol('IMT342_DestinoDirecto', 'directoActivo')))
        q_deg = self.get_joints_deg()
        return {
            'ctrlstate': ctrlstate,
            'opmode': opmode,
            'execstate': execstate,
            'pcp_module': pcp_mod,
            'pcp_routine': pcp_rout,
            'rbActivo': rb_activo,
            'directoActivo': directo_activo,
            'q_deg': q_deg,
        }

    def check_ready(self):
        st = self.get_status()
        problems = []
        active_server = None
        if st['opmode'] != 'MANR':
            problems.append(f"opmode es '{st['opmode']}' (requiere 'MANR' / Manual Reducido)")
        if st['ctrlstate'] != 'motoron':
            problems.append(f"ctrlstate es '{st['ctrlstate']}' (manten apretado el gatillo trasero del FlexPendant)")
        if st['execstate'] != 'running':
            problems.append(
                f"RAPID esta '{st['execstate']}' (puntero actual: {st['pcp_module']}/{st['pcp_routine']} -> pulsa Play)"
            )
        else:
            if st['directoActivo'] == 1:
                active_server = 'directo'
            elif st['rbActivo'] == 1:
                active_server = 'lab'
            else:
                problems.append(
                    f"esperando que arranque ServidorDirecto o aceptes 'Iniciar' en ServidorPC (puntero: {st['pcp_module']}/{st['pcp_routine']})"
                )
        return st, problems, active_server

    def _validate_limits(self, target_deg):
        for i, (val, (lo, hi)) in enumerate(zip(target_deg, LIMITS_DEG), start=1):
            if not (lo <= val <= hi):
                raise ValueError(f'Eje J{i}={val:.2f} fuera de limite seguro [{lo}, {hi}]')

    @staticmethod
    def _fmt_jt(q_deg):
        s = '[' + ','.join(f'{v:.3f}' for v in q_deg) + ']'
        return f'[{s},[9E9,9E9,9E9,9E9,9E9,9E9]]'

    def _send_directo(self, current_deg, target_deg, timeout_move=45.0):
        self._validate_limits(target_deg)
        cur_ord = int(float(self._get_symbol('IMT342_DestinoDirecto', 'directoOrden')))
        new_ord = cur_ord + 1
        self._write_symbols(
            'IMT342_DestinoDirecto',
            [
                ('directoInicio', self._fmt_jt(current_deg)),
                ('directoDestino', self._fmt_jt(target_deg)),
                ('directoOrden', str(new_ord)),
            ],
        )
        print('>>> Mira el FlexPendant: si pregunta "Confirma recorrido despejado?", toca MOVER (sin soltar el gatillo)! <<<', flush=True)
        t0 = time.time()
        while time.time() - t0 < timeout_move:
            hecho = int(float(self._get_symbol('IMT342_DestinoDirecto', 'directoHecho')))
            estado = int(float(self._get_symbol('IMT342_DestinoDirecto', 'directoEstado')))
            if hecho == new_ord and estado not in (1, 3):
                desc = DIRECTO_CODE_DESC.get(estado, f'estado {estado}')
                if estado == 2:
                    return [f'ServidorDirecto orden={new_ord} -> {desc}']
                raise RuntimeError(f'ServidorDirecto rechazo orden={new_ord}: {desc} (directoEstado={estado})')
            time.sleep(0.15)
        raise TimeoutError(f'Tiempo agotado esperando confirmacion/movimiento de ServidorDirecto orden={new_ord}')

    def _send_lab_step(self, target_deg, timeout_move=20.0):
        self._validate_limits(target_deg)
        rapid_jt = self._fmt_jt(target_deg)
        self._write_symbols('LabJointRws', [('pcTarget', rapid_jt)])
        readback = self._get_symbol('LabJointRws', 'pcTarget')
        nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', readback)][:6]
        if len(nums) != 6 or any(abs(a - b) > 0.05 for a, b in zip(nums, target_deg)):
            raise RuntimeError(f'Verificacion de pcTarget fallo: escrito={target_deg}, leido={nums}')

        cur_seq = int(float(self._get_symbol('LabJointRws', 'pcSeq')))
        new_seq = cur_seq + 1
        self._write_symbols('LabJointRws', [('pcSeq', str(new_seq))])

        t0 = time.time()
        while time.time() - t0 < timeout_move:
            rb_seq = int(float(self._get_symbol('LabJointRws', 'rbSeq')))
            rb_code = int(float(self._get_symbol('LabJointRws', 'rbCodigo')))
            if rb_seq == new_seq and rb_code != 1:
                desc = LAB_CODE_DESC.get(rb_code, f'codigo {rb_code}')
                if rb_code == 2:
                    return f'seq={new_seq} -> {desc}'
                raise RuntimeError(f'Robot rechazo orden seq={new_seq}: {desc} (rbCodigo={rb_code})')
            time.sleep(0.1)
        raise TimeoutError(f'Tiempo agotado esperando que el robot complete seq={new_seq}')

    def _try_start_rapid(self, st):
        if st.get('pcp_routine') not in ('ServidorDirecto', 'ServidorPC'):
            return False
        acquired = False
        try:
            acquired = self._request_rapid_mastership()
        except Exception:
            pass
        try:
            r = self.s.post(
                f'{self.base}/rw/rapid/execution?action=start',
                data={
                    'regain': 'continue',
                    'execmode': 'continue',
                    'cycle': 'forever',
                    'condition': 'none',
                    'stopatbp': 'disabled',
                    'alltaskbytsp': 'false',
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=self.timeout,
            )
            print(f'Intento de arranque automatico RAPID -> HTTP {r.status_code}', flush=True)
            return r.status_code in (200, 204)
        except Exception as exc:
            print(f'No se pudo arrancar RAPID remoto: {exc}', flush=True)
            return False
        finally:
            if acquired:
                self._release_rapid_mastership()

    def wait_until_ready(self, wait_sec=120.0):
        t0 = time.time()
        last_msg = ''
        while time.time() - t0 < wait_sec:
            st, problems, server = self.check_ready()
            if not problems and server is not None:
                print(f'Programa activo ({server})! Pidiendo permiso Grant si hace falta...', flush=True)
                self._ensure_rmmp(wait_sec=max(30.0, wait_sec - (time.time() - t0)))
                st, problems, server = self.check_ready()
                if not problems and server is not None:
                    print(f'Robot listo ({server})! Enviando movimiento...', flush=True)
                    return st, server
            msg = ' | '.join(problems)
            if msg != last_msg:
                print('Esperando al FlexPendant:', msg, flush=True)
                last_msg = msg
            time.sleep(0.3)
        raise TimeoutError('Tiempo agotado esperando gatillo + Play en el FlexPendant')

    def move_abs_deg(self, target_deg, max_step_deg=8.0, dry_run=False, wait_sec=0.0):
        if dry_run:
            st = self.get_status()
            current = st['q_deg']
            max_diff = max(abs(t - c) for t, c in zip(target_deg, current))
            n_steps = 1 if max_diff <= max_step_deg else int(max_diff // max_step_deg) + 1
            return [[c + (k / n_steps) * (t - c) for c, t in zip(current, target_deg)] for k in range(1, n_steps + 1)]

        if wait_sec > 0:
            st, server = self.wait_until_ready(wait_sec=wait_sec)
        else:
            st, problems, server = self.check_ready()
            if problems or server is None:
                raise RuntimeError('Robot no listo para mover:\n  - ' + '\n  - '.join(problems))

        current = st['q_deg']
        if server == 'directo':
            return self._send_directo(current, target_deg)

        max_diff = max(abs(t - c) for t, c in zip(target_deg, current))
        n_steps = 1 if max_diff <= max_step_deg else int(max_diff // max_step_deg) + 1
        results = []
        for k in range(1, n_steps + 1):
            alpha = float(k) / float(n_steps)
            wp = [c + alpha * (t - c) for c, t in zip(current, target_deg)]
            msg = self._send_lab_step(wp)
            results.append(f'paso {k}/{n_steps}: {msg}')
        return results


class RWSSendNode:
    def __init__(self):
        p = rospy.get_param
        self.client = ABBRwsMoverClient(
            robot_ip=str(p('~robot_ip', '192.168.125.1')),
            port=int(p('~rws_port', 80)),
            user=str(p('~rws_user', 'Default User')),
            password=str(p('~rws_pass', 'robotics')),
        )
        self.max_step_deg = float(p('~max_step_deg', 8.0))
        self.pub_status = rospy.Publisher('/irb120/send_status', String, queue_size=10)
        self.sub = rospy.Subscriber('/irb120/target_joints_deg', Float64MultiArray, self._on_target, queue_size=1)
        rospy.loginfo('rws_mover nodo ROS listo. Envia objetivos a /irb120/target_joints_deg (6 floats en grados).')

    def _on_target(self, msg):
        if len(msg.data) != 6:
            txt = f'ERROR: se esperaban 6 angulos en grados, llegaron {len(msg.data)}'
            rospy.logerr(txt)
            self.pub_status.publish(String(data=txt))
            return
        try:
            res = self.client.move_abs_deg(list(msg.data), max_step_deg=self.max_step_deg)
            txt = 'OK: ' + ' | '.join(res)
            rospy.loginfo(txt)
            self.pub_status.publish(String(data=txt))
        except Exception as exc:
            txt = f'ERROR: {exc}'
            rospy.logerr(txt)
            self.pub_status.publish(String(data=txt))


def main():
    try:
        _logf = open('/tmp/rws_mover_last.log', 'w')
        sys.stdout = _Tee(sys.__stdout__, _logf)
        sys.stderr = _Tee(sys.__stderr__, _logf)
    except Exception:
        pass
    args = rospy.myargv(argv=sys.argv)[1:]
    if not args:
        rospy.init_node('rws_mover_node')
        RWSSendNode()
        rospy.spin()
        return

    parser = argparse.ArgumentParser(description='Enviar movimiento seguro a ABB IRB 120 (ServidorDirecto o LabJointRws)')
    parser.add_argument('--ip', default='192.168.125.1')
    parser.add_argument('--status', action='store_true', help='Solo consultar estado actual del robot')
    parser.add_argument('--abs', nargs=6, type=float, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'), help='Objetivo absoluto en grados')
    parser.add_argument('--delta', nargs=6, type=float, metavar=('D1', 'D2', 'D3', 'D4', 'D5', 'D6'), help='Desplazamiento relativo en grados')
    parser.add_argument('--dry-run', action='store_true', help='Mostrar pasos sin enviar al robot')
    parser.add_argument('--wait', action='store_true', help='Esperar hasta 120s a que pulses Grant + gatillo + Play')
    parsed = parser.parse_args(args)

    client = ABBRwsMoverClient(robot_ip=parsed.ip)
    if parsed.status:
        st, problems, server = client.check_ready()
        print('Estado IRC5:', st)
        if problems:
            print('Falta para poder mover:')
            for pr in problems:
                print(' -', pr)
        else:
            print(f'LISTO PARA RECIBIR MOVIMIENTOS ({server})!')
        return

    st = client.get_status()
    if parsed.abs is not None:
        target = list(parsed.abs)
    elif parsed.delta is not None:
        target = [c + d for c, d in zip(st['q_deg'], parsed.delta)]
    else:
        parser.print_help()
        return

    print('Actual (deg):', [round(x, 3) for x in st['q_deg']])
    print('Objetivo (deg):', [round(x, 3) for x in target])
    res = client.move_abs_deg(target, dry_run=parsed.dry_run, wait_sec=120.0 if parsed.wait else 0.0)
    print('Resultado:', res)


if __name__ == '__main__':
    main()
