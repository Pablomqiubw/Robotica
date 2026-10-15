#!/usr/bin/env python3
"""
send_test_pose.py  -  Publica una pose OBJETIVO al nodo ik_lspb_node (solo para pruebas en simulacion).

Uso:
  rosrun irb120_description send_test_pose.py                      # pose de ejemplo
  rosrun irb120_description send_test_pose.py q1 q2 q3 q4 q5 q6    # pose = FK(q)  (alcanzable por construccion)

La pose se calcula con la FK del Hito 1, asi que es alcanzable y de ella se conoce el q "verdadero".
"""
import os
import sys
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from irb120_fk import forward_kinematics
from irb120_planner import R_to_quat

if __name__ == '__main__':
    rospy.init_node('send_test_pose')
    args = [a for a in sys.argv[1:] if not a.startswith('__')]     # ignora args internos de ROS
    q = np.array([float(a) for a in args]) if len(args) == 6 else np.array([0.4, -0.3, 0.5, 0.3, 0.8, -0.2])
    T = forward_kinematics(q)
    quat = R_to_quat(T[:3, :3])
    pub = rospy.Publisher('irb120/target_pose', PoseStamped, queue_size=1, latch=True)
    msg = PoseStamped()
    msg.header.frame_id = rospy.get_param('~base_frame', 'base_link')
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(v) for v in T[:3, 3]]
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = [float(v) for v in quat]
    rospy.sleep(1.0)                                              # tiempo para que se conecte el suscriptor
    pub.publish(msg)
    rospy.loginfo(f"q usado = {np.round(q, 4)}")
    rospy.loginfo(f"pose publicada: xyz = {np.round(T[:3, 3], 4)}  quat(xyzw) = {np.round(quat, 4)}")
    rospy.sleep(1.0)
