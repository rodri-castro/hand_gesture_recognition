#!/usr/bin/env python3
import os
import rospy
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker

# Importar custom msgs
from hand_gesture_recognition.msg import KeyPoint2DArray, KeyPoint3DArray, KeyPoint3D

# Import the 3D skeleton tracker from skeleton_3d ROS package
from skeleton_3d.skeleton_tracker_3d_ext import SkeletonTracker3DExt

# Convierte una lista de KeyPoint2D a un array numpy de keypoints (los 21 puntos)
def process_KeyPoint2DArray_msg(hand_skeleton_list, keypoints_names):
    """
    Convierte una lista de mensajes KeyPoint2D (cada uno con los 21 puntos de la mano)
    en un array numpy de shape (21, 3), donde cada fila es [x, y, 1.0] para cada keypoint.

    Args:
        hand_skeleton_list [KeyPoint2DArray.data]: lista de mensajes KeyPoint2D. Pueden venir más de una mano derecha o izda o ambas
        keypoints_names: lista de nombres de keypoints que se quieren extraer

    Returns:
        np.ndarray: array de shape (21, 3) con las coordenadas [x, y, 1.0] de cada keypoint.
                    Si falta algún punto, se rellena con [0.0, 0.0, 0.0].
    """

    skeletons = []

    for idx, skeleton in enumerate(hand_skeleton_list): # Recorre cada indice de KeyPoint2DArray. Vienen varios esqueletos de manos.

        if skeleton.name == "Right": # Solo selecciona los esqueletos de manos derechas. TODO: No dejar pasar más de uno.
            skeleton_row = []
            for name in keypoints_names: # Recorre los skeleton del Keypoint2D
                kp = getattr(skeleton, name, None)
                if kp is not None:
                    skeleton_row.append([kp.x, kp.y, 1.0])
                else:
                    skeleton_row.append([0.0, 0.0, 0.0])  # Valor por defecto si falta el punto
            skeletons.append(skeleton_row)

    if skeletons:
        # rospy.logdebug(f"Array numpy generado con shape {np.array(skeletons[0]).shape}")
        return np.array(skeletons[0]) # Retorna el primer esqueleto encontrado, si hay más de uno
    else:
        # rospy.logdebug("No se encontraron esqueletos, devolviendo array de ceros.")
        return np.zeros((len(keypoints_names), 3))

class TriangulationNode:
    def __init__(self):
        rospy.init_node('triangulate_kp_node', log_level=rospy.INFO)  # o rospy.DEBUG, rospy.INFO
        rospy.loginfo("Nodo de triangulación iniciado.")

        # Obtener los topics de los parámetros del launch file
        cartesian_detection_topic_cam3 = rospy.get_param("~cartesian_pub_topic_camera_3", "/camera3/hand/cartesian_detection")
        cartesian_detection_topic_cam4 = rospy.get_param("~cartesian_pub_topic_camera_4", "/camera4/hand/cartesian_detection")

        self.pub = rospy.Publisher("/hand_pose_3d", KeyPoint3DArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher("/hand_pose_3d_pointcloud", PointCloud, queue_size=10)
        self.marker_pub = rospy.Publisher("hand_pose_3d_markers", Marker, queue_size=1)

        # Puntos a triangular
        if rospy.has_param("~kp_to_triangulate"):
            kp_to_triangulate = rospy.get_param("~kp_to_triangulate")
            if kp_to_triangulate == "reduced":
                self.keypoints = ["wrist", "index_finger_mcp", "pinky_mcp"]
            elif kp_to_triangulate == "all":
                self.keypoints = [
                    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
                    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
                    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
                    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
                ]
            else:
                rospy.logwarn(f"Valor desconocido para kp_to_triangulate: {kp_to_triangulate}. Usando puntos reducidos.")
                self.keypoints = ["wrist", "index_finger_mcp", "pinky_mcp"]
        else:
            rospy.logwarn("No se encontró el parámetro kp_to_triangulate, usando puntos reducidos por defecto.")
            self.keypoints = ["wrist", "index_finger_mcp", "pinky_mcp"]

    
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


        sub3 = Subscriber(cartesian_detection_topic_cam3, KeyPoint2DArray)
        sub4 = Subscriber(cartesian_detection_topic_cam4, KeyPoint2DArray)

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

        self.flag = None # Flag para controlar el log del estado del nodo


    def sync_callback(self, msg3, msg4):
        # NOTA:
        # msgX.data contiene todos los esqueletos detectados de la cámara X clasificados como right o left

        # rospy.logdebug("Mensajes sincronizados recibidos.")

        if not msg3.data or not msg4.data:

            if self.flag == None or self.flag == False:
                rospy.logwarn("Algún mensaje no contiene esqueletos, se detiene la triangulación.")
                self.flag = True

            # Publicar mensajes nulos para marker y pointcloud
            empty_marker = Marker()
            empty_marker.header = msg3.header
            empty_marker.header.frame_id = "base_link"
            empty_marker.ns = "hand_connections"
            empty_marker.id = 0
            empty_marker.type = Marker.LINE_LIST
            empty_marker.action = Marker.DELETE
            self.marker_pub.publish(empty_marker)

            empty_pointcloud = PointCloud()
            empty_pointcloud.header = msg3.header
            empty_pointcloud.header.frame_id = "base_link"
            empty_pointcloud.points = []
            self.pointcloud_pub.publish(empty_pointcloud)

            return

        # Convertir KeyPoint2DArray a numpy arrays
        skeletonsX = [
            process_KeyPoint2DArray_msg(msg3.data, self.all_keypoints),
            process_KeyPoint2DArray_msg(msg4.data, self.all_keypoints)
        ]
        
        # rospy.logdebug(f"Shapes de skeletonsX: {[skel.shape for skel in skeletonsX]}")

        # Imprimir los arrays skeletonsX para depuración
        # rospy.logdebug(f"skeletonsX: {skeletonsX}")

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
            kp3d_array_msg.keypoints.append(kp3d)

        # rospy.logdebug("Publicando kp 3D triangulados.")

        if self.flag == None or self.flag == True:
            rospy.loginfo("Triangulando...")
            self.flag = False

        self.pub.publish(kp3d_array_msg)

        # Publicar como nube de puntos para RVIZ
        pointcloud_msg = PointCloud()
        pointcloud_msg.header = kp3d_array_msg.header
        pointcloud_msg.points = [
            Point(kp3d.point.x, kp3d.point.y, kp3d.point.z) for kp3d in kp3d_array_msg.keypoints
        ]
        self.pointcloud_pub.publish(pointcloud_msg)
        
        PLOT_CONNECTORS = True

        if PLOT_CONNECTORS:

            # Define las conexiones por pares de índices
            connections = [
                [0, 1, 2, 3, 4],
                [0, 5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [0, 17, 18, 19, 20],
                [5, 9, 13, 17]
            ]

            # Convierte las conexiones en pares de puntos para LINE_LIST
            line_points = []
            for conn in connections:
                for i in range(len(conn) - 1):
                    p1 = kp3d_array_msg.keypoints[conn[i]].point
                    p2 = kp3d_array_msg.keypoints[conn[i+1]].point
                    pt1 = Point(x=p1.x, y=p1.y, z=p1.z)
                    pt2 = Point(x=p2.x, y=p2.y, z=p2.z)
                    line_points.extend([pt1, pt2])

            marker = Marker()
            marker.header = kp3d_array_msg.header
            marker.ns = "hand_connections"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.01  # Grosor de la línea
            marker.color.r = 1.0
            marker.color.g = 0.75
            marker.color.b = 0.13
            marker.color.a = 1.0
            marker.points = line_points
            marker.lifetime = rospy.Duration(0.2)  # Lifetime de 0.2 segundos

            self.marker_pub.publish(marker)
            
if __name__ == '__main__':
    try:
        TriangulationNode()
        rospy.loginfo("Nodo de triangulación ejecutándose. Esperando mensajes...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Cerrando nodo de triangulación interrumpido.")


