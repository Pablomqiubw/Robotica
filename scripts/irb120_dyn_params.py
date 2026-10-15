"""
irb120_dyn_params.py
Parametros INERCIALES de los 6 links.

!!! ATENCION: LOS VALORES DE ESTE ARCHIVO SON DE DEMOSTRACION (DEMO) !!!
No provienen del IRB 120 real ni de tu URDF. Sirven para validar el algoritmo
de dinamica. Para resultados validos para tu robot, reemplaza demo_link_params()
por los datos reales (bloques <inertial> del URDF o datos del fabricante).

Para cada link i (marco del link = marco justo despues de la rotacion de la junta i):
    m   : masa [kg]
    com : centro de masa expresado en el marco del link [m]  (3,)
    I   : tensor de inercia respecto al CENTRO DE MASA, en el marco del link [kg m^2] (3,3)

Regla DEMO usada: m = 1 kg; COM en el punto medio del segmento hasta la siguiente
junta; inercia isotropa de varilla delgada I = m L^2/12 * Identidad (L = longitud del segmento).
"""
import numpy as np
from irb120_params import JOINTS, GRIPPER_FIXED_XYZ

GRAVITY = np.array([0.0, 0.0, -9.81])      # m/s^2 en el frame base_link (eje z hacia arriba)


def demo_link_params():
    segs = [JOINTS[i + 1][0] for i in range(5)] + [GRIPPER_FIXED_XYZ]
    links = []
    for s in segs:
        s = np.array(s, dtype=float)
        L = max(np.linalg.norm(s), 0.05)
        m = 1.0
        links.append({'m': m, 'com': 0.5 * s, 'I': (m * L ** 2 / 12.0) * np.eye(3)})
    return links
