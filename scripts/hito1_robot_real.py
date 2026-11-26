#!/usr/bin/env python3
"""
hito1_robot_real.py - Aplicacion del Hito 1 (Cinematica Directa + Sistema de Seguridad) al ABB IRB 120 real.

Flujo:
  1. Lee posicion actual del robot real por RWS (o usa q=[0..0] en --sim).
  2. Calcula ecuaciones T0^1..T5^6 y T0^6 = prod(Ti) hacia el objetivo pedido.
  3. Evalua los chequeos matematicos de seguridad (limites, alturas Z de punta/codo/muneca, anti-autocolision, pasos).
  4. Muestra todo el reporte en pantalla y PIDE PERMISO explicito al usuario antes de enviar por TCP (puerto 5005).
  5. Si el usuario autoriza, mueve el robot real y compara la posicion real final vs la calculada por el Hito 1.

Ejemplos de uso:
  rosrun irb120_description hito1_robot_real.py --abs 10 0 0 0 0 0
  rosrun irb120_description hito1_robot_real.py --delta 0 0 0 0 0 5
  rosrun irb120_description hito1_robot_real.py --sim --abs 15 -10 10 0 20 0
  rosrun irb120_description hito1_robot_real.py   (modo interactivo: pide J1..J6)
"""
import argparse
import math
import os
import sys

import requests
from requests.auth import HTTPDigestAuth

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hito1_seguridad import evaluar_seguridad, fk_detallada_deg, formatear_reporte_hito1
from tcp_mover import ABBTcpMoverClient


def read_rws_state(robot_ip='192.168.125.1', rws_port=80, user='Default User', password='robotics'):
    s = requests.Session()
    s.auth = HTTPDigestAuth(user, password)
    base = f'http://{robot_ip}:{rws_port}'
    jt = s.get(f'{base}/rw/motionsystem/mechunits/ROB_1/jointtarget?json=1', timeout=2.5).json()['_embedded']['_state'][0]
    q_deg = [float(jt[f'rax_{i}']) for i in range(1, 7)]
    rt = s.get(f'{base}/rw/motionsystem/mechunits/ROB_1/robtarget?json=1', timeout=2.5).json()['_embedded']['_state'][0]
    robtarget_mm = (float(rt['x']), float(rt['y']), float(rt['z']))
    return q_deg, robtarget_mm


def ask_interactive_joints(current_deg):
    print('\nPosicion actual (deg):', [round(x, 2) for x in current_deg])
    print('Ingresa los 6 angulos objetivo J1..J6 en grados separados por espacio (ej: 10 0 0 0 0 0)')
    raw = input('Objetivo [J1 J2 J3 J4 J5 J6]: ').strip()
    parts = raw.replace(',', ' ').split()
    if len(parts) != 6:
        raise ValueError(f'Se esperaban 6 numeros y se recibieron {len(parts)}.')
    return [float(x) for x in parts]


def main():
    parser = argparse.ArgumentParser(description='Hito 1 en ABB IRB 120 real con validacion matematica y permiso')
    parser.add_argument('--ip', default='192.168.125.1')
    parser.add_argument('--port', type=int, default=5005)
    parser.add_argument('--abs', nargs=6, type=float, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'), help='Objetivo absoluto en grados')
    parser.add_argument('--delta', nargs=6, type=float, metavar=('D1', 'D2', 'D3', 'D4', 'D5', 'D6'), help='Incremento relativo en grados')
    parser.add_argument('--sim', action='store_true', help='Modo simulacion/calculo sin conectar al robot real')
    parser.add_argument('--resumen', action='store_true', help='Ocultar matrices intermedias T0^1..T5^6 y mostrar solo T0^6')
    args = parser.parse_args()

    q_actual = [0.0] * 6
    rob_actual_mm = None
    if not args.sim:
        try:
            q_actual, rob_actual_mm = read_rws_state(robot_ip=args.ip)
        except Exception as exc:
            print(f'ERROR leyendo el robot en {args.ip}: {exc}')
            print('Si quieres probar solo los calculos sin robot conectado, usa --sim')
            sys.exit(1)

    if args.abs is not None:
        q_target = list(args.abs)
    elif args.delta is not None:
        q_target = [c + d for c, d in zip(q_actual, args.delta)]
    else:
        q_target = ask_interactive_joints(q_actual)

    ev = evaluar_seguridad(q_target, q_current_deg=q_actual)
    print()
    print(formatear_reporte_hito1(ev, robtarget_real_mm=rob_actual_mm, paso_a_paso=not args.resumen))

    if not ev['aprobado']:
        print('\nMovimiento cancelado automaticamente por el sistema de seguridad.')
        sys.exit(2)

    if args.sim:
        print('\n[Modo --sim] Calculo terminado. No se envio nada al robot real.')
        return

    resp = input('\n¿Autoriza mover el robot real a esta posicion? Escriba "si" para confirmar: ').strip().lower()
    if resp not in ('s', 'si', 'sí', 'y', 'yes'):
        print('Permiso NO otorgado. El robot no se movio.')
        return

    client = ABBTcpMoverClient(robot_ip=args.ip, tcp_port=args.port)
    print('\nPermiso concedido. Esperando gatillo trasero + Play (ServidorTCP) en FlexPendant...')
    res = client.move_abs_deg(q_target, max_step_deg=8.0, wait_sec=120.0)
    print('Resultado TCP:', res)

    try:
        q_final, rob_final_mm = read_rws_state(robot_ip=args.ip)
        fk_pred = ev['fk_target']['pos_mm']
        fk_final = fk_detallada_deg(q_final)['pos_mm']
        err_pred_vs_real = math.sqrt(
            (fk_pred[0] - rob_final_mm[0]) ** 2
            + (fk_pred[1] - rob_final_mm[1]) ** 2
            + (fk_pred[2] - rob_final_mm[2]) ** 2
        )
        print('\n--- VERIFICACION POST-MOVIMIENTO (Hito 1 vs Robot Real) ---')
        print('  q final leido (°):', [round(x, 3) for x in q_final])
        print(f'  Predicho por T0^6: ({fk_pred[0]:.2f}, {fk_pred[1]:.2f}, {fk_pred[2]:.2f}) mm')
        print(f'  FK(q_final)      : ({fk_final[0]:.2f}, {fk_final[1]:.2f}, {fk_final[2]:.2f}) mm')
        print(f'  ABB robtarget    : ({rob_final_mm[0]:.2f}, {rob_final_mm[1]:.2f}, {rob_final_mm[2]:.2f}) mm')
        print(f'  Diferencia total : {err_pred_vs_real:.2f} mm')
    except Exception as exc:
        print(f'Movimiento ejecutado, no se pudo leer verificacion final: {exc}')


if __name__ == '__main__':
    main()
