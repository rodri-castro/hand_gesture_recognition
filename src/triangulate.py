#!/usr/bin/env python3
import os
import rospy
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud

# Importar custom msgs
from hand_gesture_recognition.msg import KeyPoint2DArray, KeyPoint3DArray, KeyPoint3D

# Import the 3D skeleton tracker from skeleton_3d ROS package
from skeleton_3d.skeleton_tracker_3d_ext import SkeletonTracker3DExt

# Convierte una lista de KeyPoint2D a un array numpy de keypoints (los 21 puntos)
def kp2darray_to_np(kp_list, keypoints_names):
    """
    Convierte una lista de mensajes KeyPoint2D (cada uno con los 21 puntos de la mano)
    en un array numpy de shape (21, 3), donde cada fila es [x, y, 1.0] para cada keypoint.

    Args:
        kp_list: lista de mensajes KeyPoint2D (usualmente de longitud 1, una mano por frame)
        keypoints_names: lista de nombres de keypoints (ordenados)

    Returns:
        np.ndarray: array de shape (21, 3) con las coordenadas [x, y, 1.0] de cada keypoint.
                    Si falta algún punto, se rellena con [0.0, 0.0, 0.0].
    """
    rospy.logdebug(f"Convirtiendo {len(kp_list)} elementos de KeyPoint2D a numpy array.")
    keypoints = []
    for idx, kp in enumerate(kp_list):
        kp_row = []
        # rospy.loginfo(f"Procesando keypoint {idx}: {kp}")
        for name in keypoints_names:
            pt = getattr(kp, name, None)
            if pt is not None:
                # rospy.loginfo(f"  {name}: ({pt.x}, {pt.y})")
                kp_row.append([pt.x, pt.y, 1.0])
            else:
                # rospy.loginfo(f"  {name}: no encontrado, usando [0.0, 0.0, 0.0]")
                kp_row.append([0.0, 0.0, 0.0])  # Valor por defecto si falta el punto
        keypoints.append(kp_row)
    if keypoints:
        # rospy.loginfo(f"Array numpy generado con shape {np.array(keypoints[0]).shape}")
        return np.array(keypoints[0])
    else:
        rospy.logdebug("No se encontraron keypoints, devolviendo array de ceros.")
        return np.zeros((len(keypoints_names), 3))

class TriangulationNode:
    def __init__(self):
        rospy.init_node('triangulate_kp_node')
        rospy.loginfo("Nodo de triangulación iniciado.")

        self.pub = rospy.Publisher("/triangulated_hand_kp_array", KeyPoint3DArray, queue_size=10)

        # Puntos a triangular
        self.keypoints = rospy.get_param("~keypoints", ["wrist", "index_finger_mcp", "pinky_mcp"])
        rospy.loginfo(f"Puntos clave a triangular: {self.keypoints}")

        # Load calibration matrices
        base_path = rospy.get_param("~calib_path", "~/rosWorkspace/ros_ws/src/skeleton_3d")
        base_path = os.path.expanduser(base_path)
        rospy.loginfo(f"Ruta base de calibración: {base_path}")

        # Load transformation matrices and intrinsic parameters for each camera
        self.geometrics = []
        self.intrinsics = []

        for cam in range(3, 4+1): # 1-based index
            geometric_path = f'{base_path}/config/camera{cam}_geometric.txt'
            intrinsic_path = f'{base_path}/config/camera{cam}_intrinsic.txt'
            if not os.path.exists(geometric_path) or not os.path.exists(intrinsic_path):
                msg = f'Missing configuration files for camera {cam}. Expected: {geometric_path} and {intrinsic_path}.'
                rospy.logerr(msg)
                rospy.signal_shutdown(msg)
            else:
                self.geometrics.append(np.loadtxt(geometric_path))
                self.intrinsics.append(np.loadtxt(intrinsic_path))
                rospy.loginfo("Matrices de calibración cargadas correctamente.")


        sub3 = Subscriber("/camera3/hand_cartesian_detections", KeyPoint2DArray)
        sub4 = Subscriber("/camera4/hand_cartesian_detections", KeyPoint2DArray)

        ats = ApproximateTimeSynchronizer([sub3, sub4], queue_size=10, slop=0.1)
        ats.registerCallback(self.sync_callback)
        rospy.loginfo("Subscriptores y sincronizador configurados.")

        # Parámetros del tracker 3D
        self.num_keypoints = 21
        self.num_cameras = 2
        self.max_error = 0.05  # Error máxima de triangulación (en metros)
        self.max_distance = 5.0  # Distancia máxima de seguimiento (en metros)
        self.num_frames = 10  # Número de frames para el seguimiento
        self.use_kalman = True  # Usar filtro de Kalman

        # iniciar tracker 3d
        self.tracker = SkeletonTracker3DExt(
            num_keypoints = self.num_keypoints,
            num_cameras = self.num_cameras,
            intrinsics = self.intrinsics,
            geometrics = self.geometrics,
            max_error = self.max_error, # triangulation error (meters)
            max_distance = self.max_distance, # tracking maximum distance (meters)
            num_frames = self.num_frames,
            use_kalman = self.use_kalman
        )

        # Lista completa de keypoints en orden
        self.all_keypoints = [
            "wrist",
            "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
            "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
            "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]
        self.num_keypoints = len(self.all_keypoints)


    def sync_callback(self, msg3, msg4):
        rospy.logdebug("Mensajes sincronizados recibidos.")
        if not msg3.data or not msg4.data:
            rospy.logwarn("Algún mensaje no contiene datos, se omite la triangulación.")
            return

        # Convertir KeyPoint2DArray a numpy arrays
        rospy.logdebug("Convirtiendo KeyPoint2DArray a numpy arrays para ambas cámaras.")
        skeletonsX = [
            kp2darray_to_np(msg3.data, self.all_keypoints),
            kp2darray_to_np(msg4.data, self.all_keypoints)
        ]
        
        rospy.logdebug(f"Shapes de skeletonsX: {[skel.shape for skel in skeletonsX]}")

        # Imprimir los arrays skeletonsX para depuración
        rospy.logdebug(f"skeletonsX: {skeletonsX}")

        # Si alguno de los arrays está vacío, omitir
        if any(skel.shape[0] == 0 for skel in skeletonsX):
            rospy.logwarn("Algún esqueleto no tiene keypoints, se omite la triangulación.")
            return

        try:
            keypoints_3D = self.tracker.track_skeleton(skeletonsX)
            # rospy.loginfo(f"keypoints_3D shape: {keypoints_3D.shape}")
            # rospy.loginfo(f"keypoints_3D: {keypoints_3D}")
        except Exception as e:
            rospy.logerr(f"Error durante la triangulación 3D: {e}")
            return

        # Convert keypoints to KeyPoint3DArray message
        kp3d_array_msg = KeyPoint3DArray()
        kp3d_array_msg.header = msg3.header
        kp3d_array_msg.header.frame_id = "base_link"
        kp3d_array_msg.keypoints = []

        # Publica solo los puntos definidos en self.keypoints
        for name in self.keypoints:
            i = self.all_keypoints.index(name)
            kp3d = KeyPoint3D()
            kp3d.name = name
            kp3d.point.x = keypoints_3D[i, 0]
            kp3d.point.y = keypoints_3D[i, 1]
            kp3d.point.z = keypoints_3D[i, 2]
            # rospy.logdebug(f"Publicando keypoint {name}: ({kp3d.point.x}, {kp3d.point.y}, {kp3d.point.z})")
            kp3d_array_msg.keypoints.append(kp3d)

        rospy.logdebug("Publicando kp 3D triangulados.")
        self.pub.publish(kp3d_array_msg)

        # Publicar como nube de puntos para RVIZ
        pointcloud_pub = getattr(self, 'pointcloud_pub', None)
        if pointcloud_pub is None:
            self.pointcloud_pub = rospy.Publisher("/triangulated_hand_pointcloud", PointCloud, queue_size=10)
            pointcloud_pub = self.pointcloud_pub

        pointcloud_msg = PointCloud()
        pointcloud_msg.header = kp3d_array_msg.header
        pointcloud_msg.points = [
            Point(kp3d.point.x, kp3d.point.y, kp3d.point.z) for kp3d in kp3d_array_msg.keypoints
        ]
        pointcloud_pub.publish(pointcloud_msg)


if __name__ == '__main__':
    try:
        TriangulationNode()
        rospy.loginfo("Nodo de triangulación ejecutándose. Esperando mensajes...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Cerrando nodo de triangulación interrumpido.")


