#!/usr/bin/env python3
"""
hito1_seguridad.py - Cinematica Directa (Hito 1) y Sistema de Seguridad para el ABB IRB 120 real.

Parametros geometricos verificados del ABB IRB 120-3/0.6 (coinciden con abb_irb120_support y el controlador IRC5):
  T0^1(q1) = Trans(0, 0, 0.000) * Rz(q1)
  T1^2(q2) = Trans(0, 0, 0.290) * Ry(q2)
  T2^3(q3) = Trans(0, 0, 0.270) * Ry(q3)
  T3^4(q4) = Trans(0, 0, 0.070) * Rx(q4)
  T4^5(q5) = Trans(0.302, 0, 0.000) * Ry(q5)
  T5^6(q6) = Trans(0.072, 0, 0.000) * Rx(q6)
  T0^6(q)  = T0^1 * T1^2 * T2^3 * T3^4 * T4^5 * T5^6
"""
import math
import numpy as np


# Limites del ABB IRB 120 con 2 deg de margen de seguridad (coinciden con ROS_TcpMover.mod)
JOINT_LIMITS_DEG = [
    (-163.0, 163.0),   # J1
    (-108.0, 108.0),   # J2
    (-108.0, 68.0),    # J3
    (-158.0, 158.0),   # J4
    (-118.0, 118.0),   # J5
    (-398.0, 398.0),   # J6
]

# Longitudes reales en metros (ABB IRB 120-3/0.6)
D1 = 0.290
L2 = 0.270
A3 = 0.070
D4 = 0.302
D6 = 0.072

# Umbrales de seguridad de celda (base_link esta sobre la estructura azul)
Z_MIN_TIP_MM = 50.0     # Altura minima segura para la punta (mm)
Z_MIN_ELBOW_MM = 80.0   # Altura minima segura para codo/muneca (mm)
MIN_BASE_RADIUS_MM = 110.0  # Radio minimo respecto al eje base cuando Z < 420 mm
MIN_BASE_CHECK_Z_MM = 420.0
MAX_ARM_REACH_FROM_SHOULDER_MM = 582.0  # Max distancia fisica brazo desde hombro (270+310)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def homog(R, xyz):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def rot_to_rpy_deg(R):
    """Extrae Roll, Pitch, Yaw (convencion ZYX) en grados."""
    sy = -R[2, 0]
    cy = math.hypot(R[0, 0], R[1, 0])
    if cy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(sy, cy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(sy, cy)
        yaw = 0.0
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def fk_detallada_deg(q_deg):
    """
    Calcula todas las matrices individuales T_{i-1}^i, acumuladas T_0^i,
    puntos clave de eslabones (hombro, codo, muneca, brida) y pose final.
    """
    q_rad = [math.radians(float(v)) for v in q_deg]

    specs = [
        ('T0^1', [0.0, 0.0, 0.0], 'z', q_rad[0]),
        ('T1^2', [0.0, 0.0, D1], 'y', q_rad[1]),
        ('T2^3', [0.0, 0.0, L2], 'y', q_rad[2]),
        ('T3^4', [0.0, 0.0, A3], 'x', q_rad[3]),
        ('T4^5', [D4, 0.0, 0.0], 'y', q_rad[4]),
        ('T5^6', [D6, 0.0, 0.0], 'x', q_rad[5]),
    ]

    rot_fn = {'x': rx, 'y': ry, 'z': rz}
    locales = []
    acumuladas = []
    T = np.eye(4, dtype=float)
    for nombre, xyz, axis, qi in specs:
        Ti = homog(rot_fn[axis](qi), xyz)
        T = T @ Ti
        locales.append((nombre, axis, qi, xyz, Ti))
        acumuladas.append(T.copy())

    # Puntos importantes en metros respecto a base_link
    p_base = np.zeros(3)
    p_hombro = acumuladas[1][:3, 3]
    p_codo = acumuladas[2][:3, 3]
    p_muneca = acumuladas[4][:3, 3]
    p_brida = acumuladas[5][:3, 3]

    return {
        'q_deg': [float(v) for v in q_deg],
        'q_rad': q_rad,
        'locales': locales,
        'acumuladas': acumuladas,
        'T0_6': acumuladas[5],
        'pos_m': p_brida,
        'pos_mm': p_brida * 1000.0,
        'rpy_deg': rot_to_rpy_deg(acumuladas[5][:3, :3]),
        'puntos_mm': {
            'base': p_base * 1000.0,
            'hombro_J2': p_hombro * 1000.0,
            'codo_J3': p_codo * 1000.0,
            'muneca_J5': p_muneca * 1000.0,
            'brida_J6': p_brida * 1000.0,
        },
    }


def evaluar_seguridad(q_target_deg, q_current_deg=None, z_min_tip_mm=Z_MIN_TIP_MM, z_min_elbow_mm=Z_MIN_ELBOW_MM):
    """
    Evalua todas las condiciones matematicas y geometricas de seguridad del movimiento.
    Retorna un diccionario con 'aprobado' (bool), 'checks' (lista de dicts) y 'fk_target'.
    """
    fk_tgt = fk_detallada_deg(q_target_deg)
    checks = []

    # 1. Limites articulares con margen de 2 deg
    lim_bad = []
    for i, (val, (lo, hi)) in enumerate(zip(q_target_deg, JOINT_LIMITS_DEG), start=1):
        if not (lo <= val <= hi):
            lim_bad.append(f'J{i}={val:.2f}° fuera de [{lo:.0f}°, {hi:.0f}°]')
    checks.append({
        'nombre': '1. Limites articulares (margen 2°)',
        'ok': len(lim_bad) == 0,
        'detalle': 'Todas las juntas dentro de rango seguro' if not lim_bad else '; '.join(lim_bad),
    })

    # 2. Altura minima sobre la mesa / estructura azul (codo, muneca y punta)
    pts = fk_tgt['puntos_mm']
    z_tip = float(pts['brida_J6'][2])
    z_mun = float(pts['muneca_J5'][2])
    z_cod = float(pts['codo_J3'][2])
    z_ok = (z_tip >= z_min_tip_mm) and (z_mun >= z_min_elbow_mm) and (z_cod >= z_min_elbow_mm)
    checks.append({
        'nombre': '2. Altura segura sobre mesa/marco',
        'ok': z_ok,
        'detalle': (
            f'Z_punta={z_tip:.1f} mm (>={z_min_tip_mm:.0f}), '
            f'Z_muneca={z_mun:.1f} mm, Z_codo={z_cod:.1f} mm (>={z_min_elbow_mm:.0f})'
        ),
    })

    # 3. Envolvente / anti-autocolision con la columna base
    r_xy_tip = float(math.hypot(pts['brida_J6'][0], pts['brida_J6'][1]))
    r_xy_mun = float(math.hypot(pts['muneca_J5'][0], pts['muneca_J5'][1]))
    dist_hombro = float(np.linalg.norm(pts['muneca_J5'] - pts['hombro_J2']))
    autocol = (z_tip < MIN_BASE_CHECK_Z_MM and r_xy_tip < MIN_BASE_RADIUS_MM) or (
        z_mun < MIN_BASE_CHECK_Z_MM and r_xy_mun < MIN_BASE_RADIUS_MM
    )
    reach_ok = dist_hombro <= MAX_ARM_REACH_FROM_SHOULDER_MM
    checks.append({
        'nombre': '3. Anti-autocolision y alcance fisico',
        'ok': (not autocol) and reach_ok,
        'detalle': f'Radio XY punta={r_xy_tip:.1f} mm | Dist. hombro->muneca={dist_hombro:.1f} mm (max {MAX_ARM_REACH_FROM_SHOULDER_MM:.0f})',
    })

    # 4. Paso articular y desplazamiento cartesiano desde posicion actual
    fk_cur = None
    max_dq = 0.0
    n_pasos = 1
    dist_cart = 0.0
    if q_current_deg is not None:
        fk_cur = fk_detallada_deg(q_current_deg)
        dq = [abs(t - c) for t, c in zip(q_target_deg, q_current_deg)]
        max_dq = max(dq)
        n_pasos = 1 if max_dq <= 8.0 else int(max_dq // 8.0) + 1
        dist_cart = float(np.linalg.norm(fk_tgt['pos_mm'] - fk_cur['pos_mm']))
        checks.append({
            'nombre': '4. Desplazamiento desde pose actual',
            'ok': True,
            'detalle': f'Max delta articular={max_dq:.2f}° ({n_pasos} paso(s) <= 8°) | Delta cartesiano={dist_cart:.1f} mm',
        })

    aprobado = all(c['ok'] for c in checks)
    return {
        'aprobado': aprobado,
        'checks': checks,
        'fk_target': fk_tgt,
        'fk_current': fk_cur,
        'max_dq_deg': max_dq,
        'n_pasos': n_pasos,
        'dist_cart_mm': dist_cart,
    }


def formatear_reporte_hito1(eval_res, robtarget_real_mm=None, paso_a_paso=True):
    """Genera un informe legible con ecuaciones, matrices T_0^i, destino y estado de seguridad."""
    fk_t = eval_res['fk_target']
    q_deg = fk_t['q_deg']
    q_rad = fk_t['q_rad']
    lines = []
    lines.append('=' * 74)
    lines.append('  HITO 1 EN ROBOT REAL - CINEMATICA DIRECTA Y VALIDACION DE SEGURIDAD')
    lines.append('=' * 74)
    lines.append('Ecuaciones del modelo ABB IRB 120-3/0.6 (base_link -> brida tool0):')
    lines.append('  T0^6(q) = T0^1(q1) * T1^2(q2) * T2^3(q3) * T3^4(q4) * T4^5(q5) * T5^6(q6)')
    lines.append('  donde:')
    lines.append('    T0^1(q1) = Trans(0, 0, 0.000) * Rz(q1)')
    lines.append('    T1^2(q2) = Trans(0, 0, 0.290) * Ry(q2)')
    lines.append('    T2^3(q3) = Trans(0, 0, 0.270) * Ry(q3)')
    lines.append('    T3^4(q4) = Trans(0, 0, 0.070) * Rx(q4)')
    lines.append('    T4^5(q5) = Trans(0.302, 0, 0) * Ry(q5)')
    lines.append('    T5^6(q6) = Trans(0.072, 0, 0) * Rx(q6)')
    lines.append('')
    lines.append('Objetivo articular pedido:')
    lines.append('  q_deg = [' + ', '.join(f'{v:7.3f}°' for v in q_deg) + ']')
    lines.append('  q_rad = [' + ', '.join(f'{v:7.4f}' for v in q_rad) + ']')

    if eval_res['fk_current'] is not None:
        fk_c = eval_res['fk_current']
        pc = fk_c['pos_mm']
        lines.append('')
        lines.append('Posicion ACTUAL del robot real:')
        lines.append('  q_act = [' + ', '.join(f'{v:7.3f}°' for v in fk_c['q_deg']) + ']')
        lines.append(f'  FK(q_act) -> X={pc[0]:7.2f} mm, Y={pc[1]:7.2f} mm, Z={pc[2]:7.2f} mm')
        if robtarget_real_mm is not None:
            rx_mm, ry_mm, rz_mm = robtarget_real_mm
            err_mm = math.sqrt((pc[0] - rx_mm) ** 2 + (pc[1] - ry_mm) ** 2 + (pc[2] - rz_mm) ** 2)
            lines.append(
                f'  ABB robtarget real -> X={rx_mm:7.2f} mm, Y={ry_mm:7.2f} mm, Z={rz_mm:7.2f} mm '
                f'(diferencia modelo vs real = {err_mm:.2f} mm)'
            )

    if paso_a_paso:
        lines.append('')
        lines.append('Matrices individuales T_{i-1}^i evaluadas:')
        for nombre, axis, qi, xyz, Ti in fk_t['locales']:
            lines.append(f'  {nombre} (eje {axis}, q={math.degrees(qi):.3f}°={qi:.4f} rad, offset={xyz} m):')
            for row in Ti:
                lines.append('    [ ' + '  '.join(f'{val:8.4f}' for val in row) + ' ]')

    lines.append('')
    lines.append('Matriz Homogenea Total T0^6 = T0^1*T1^2*T2^3*T3^4*T4^5*T5^6:')
    for row in fk_t['T0_6']:
        lines.append('  [ ' + '  '.join(f'{val:9.5f}' for val in row) + ' ]')

    pt = fk_t['pos_mm']
    r_deg, p_deg, y_deg = fk_t['rpy_deg']
    lines.append('')
    lines.append('A DONDE VA A IR EL ROBOT (calculado por Hito 1):')
    lines.append(f'  Posicion punta (X, Y, Z) : ({pt[0]:7.2f}, {pt[1]:7.2f}, {pt[2]:7.2f}) mm')
    lines.append(f'  Orientacion RPY (ZYX)    : (Roll={r_deg:6.2f}°, Pitch={p_deg:6.2f}°, Yaw={y_deg:6.2f}°)')
    pts = fk_t['puntos_mm']
    lines.append(
        f'  Codo J3=({pts["codo_J3"][0]:.1f}, {pts["codo_J3"][1]:.1f}, {pts["codo_J3"][2]:.1f}) mm | '
        f'Muneca J5=({pts["muneca_J5"][0]:.1f}, {pts["muneca_J5"][1]:.1f}, {pts["muneca_J5"][2]:.1f}) mm'
    )

    lines.append('')
    lines.append('VERIFICACION DEL SISTEMA DE SEGURIDAD:')
    for chk in eval_res['checks']:
        tag = 'OK' if chk['ok'] else 'BLOQUEADO'
        lines.append(f'  [{tag}] {chk["nombre"]}: {chk["detalle"]}')

    lines.append('-' * 74)
    if eval_res['aprobado']:
        lines.append('  VEREDICTO: POSICION SEGURA Y VALIDA — LISTO PARA PEDIR PERMISO')
    else:
        lines.append('  VEREDICTO: MOVIMIENTO BLOQUEADO POR SEGURIDAD (NO SE ENVIARA)')
    lines.append('=' * 74)
    return '\n'.join(lines)


if __name__ == '__main__':
    # Verificacion rapida con la lectura real del robot del 2026-11-17
    q_real = [0.010982, -0.048810, 0.093123, 0.066370, -0.073861, -0.033751]
    ev = evaluar_seguridad([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], q_current_deg=q_real)
    print(formatear_reporte_hito1(ev, robtarget_real_mm=(373.824, 0.0715, 629.8035), paso_a_paso=True))
