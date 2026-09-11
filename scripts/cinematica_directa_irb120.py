#!/usr/bin/env python3
import numpy as np

def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def homog(R, p):
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T

def joint_transform(origin_xyz, origin_rpy, axis, q):
    roll, pitch, yaw = origin_rpy
    R_fixed = rotz(yaw) @ roty(pitch) @ rotx(roll)
    T_fixed = homog(R_fixed, np.array(origin_xyz))
    if axis == 'x':
        R_var = rotx(q)
    elif axis == 'y':
        R_var = roty(q)
    else:
        R_var = rotz(q)
    T_var = homog(R_var, np.array([0, 0, 0]))
    return T_fixed @ T_var

JOINTS = [
    ([0, 0, 0.05],  [0, 0, 0],       'z'),
    ([0, 0, 0.29],  [0, -1.5708, 0], 'y'),
    ([0.27, 0, 0],  [0, 1.5708, 0],  'y'),
    ([0.07, 0, 0],  [0, -1.5708, 0], 'z'),
    ([0, 0, 0.302], [0, 0, 0],       'y'),
    ([0, 0, 0.04],  [0, 0, 0],       'z'),
]
GRIPPER_FIXED_XYZ = [0, 0, 0.072]
GRIPPER_FIXED_RPY = [0, 0, 0]

def forward_kinematics(q):
    T = np.eye(4)
    for i, (xyz, rpy, axis) in enumerate(JOINTS):
        T = T @ joint_transform(xyz, rpy, axis, q[i])
    R_fixed = rotz(GRIPPER_FIXED_RPY[2]) @ roty(GRIPPER_FIXED_RPY[1]) @ rotx(GRIPPER_FIXED_RPY[0])
    T = T @ homog(R_fixed, np.array(GRIPPER_FIXED_XYZ))
    return T

def pose_from_matrix(T):
    return T[0:3, 3]

if __name__ == "__main__":
    q_test = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    T = forward_kinematics(q_test)
    pos = pose_from_matrix(T)
    print("Matriz de transformacion homogenea (base_link -> gripper):")
    print(np.round(T, 6))
    print("\nPosicion calculada [x, y, z] en metros:")
    print(pos)
