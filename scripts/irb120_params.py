"""
irb120_params.py
Parametros geometricos del modelo. UNICA fuente de verdad para FK e IK.

ORIGEN DE LOS DATOS: urdf/irb120.urdf.xacro del Hito 1 (dimensiones APROXIMADAS,
tal como indica tu resumen). NO son datos verificados del datasheet de ABB.
Cuando ajustes el URDF con el datasheet, solo cambias este archivo.

Formato de cada junta: (origin_xyz [m], origin_rpy [rad], eje_local)
"""

JOINTS = [
    ([0, 0, 0.05],  [0, 0, 0],       'z'),   # J1
    ([0, 0, 0.29],  [0, -1.5708, 0], 'y'),   # J2
    ([0.27, 0, 0],  [0, 1.5708, 0],  'y'),   # J3
    ([0.07, 0, 0],  [0, -1.5708, 0], 'z'),   # J4
    ([0, 0, 0.302], [0, 0, 0],       'y'),   # J5
    ([0, 0, 0.04],  [0, 0, 0],       'z'),   # J6
]
GRIPPER_FIXED_XYZ = [0, 0, 0.072]
GRIPPER_FIXED_RPY = [0, 0, 0]

# Limites articulares: NO se definen aqui a proposito (no inventar).
# Cargalos del URDF (<limit lower= upper=>) o del datasheet y pasalos a
# filter_by_limits(). Formato: lo = [6 valores], hi = [6 valores] en rad.
