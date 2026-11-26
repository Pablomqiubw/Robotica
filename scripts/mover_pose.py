#!/usr/bin/env python3
import os
import sys
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from irb120_fk import rpy_to_R, homog
from irb120_ik import ik_pieper


def quat_from_R(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('__')]
    if len(args) != 6:
        sys.exit("Uso: rosrun irb120_description mover_pose.py x y z roll pitch yaw  (metros y radianes)")
    x, y, z, roll, pitch, yaw = [float(v) for v in args]
    R = rpy_to_R([roll, pitch, yaw])
    T = homog(R, np.array([x, y, z]))

    sols = ik_pieper(T)
    if not sols:
        print("La pose x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f NO tiene solucion (fuera del espacio de trabajo)." %
              (x, y, z, roll, pitch, yaw))
        print("No se envio nada. Prueba otros numeros.")
        sys.exit(0)
    print("La pose tiene %d solucion(es). Enviando al nodo..." % len(sols))

    qx, qy, qz, qw = quat_from_R(R)
    rospy.init_node('mover_pose', anonymous=True)
    pub = rospy.Publisher('irb120/target_pose', PoseStamped, queue_size=1, latch=True)
    rospy.sleep(1.0)
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
    msg.pose.orientation.x, msg.pose.orientation.y = float(qx), float(qy)
    msg.pose.orientation.z, msg.pose.orientation.w = float(qz), float(qw)
    pub.publish(msg)
    rospy.sleep(1.0)
    print("Pose enviada. Mira RViz.")
