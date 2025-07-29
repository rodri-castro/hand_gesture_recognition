#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import String
from sensor_msgs.msg import Image
from gesture_recognition import *
from cvfpscalc import CvFpsCalc
from cv_bridge import CvBridge, CvBridgeError
import image_geometry
from sensor_msgs.msg import CameraInfo

from hand_gesture_recognition.msg import KeyPoint2D
from hand_gesture_recognition.msg import KeyPoint2DArray

# from sobits_msgs.srv import RunCtrl, RunCtrlResponse


class HandSignRecognition:

    def __init__(self):

        # Set params for Network
        self.keypoint_classifier_label = rospy.get_param("~keypoint_classifier_label")
        self.keypoint_classifier_model = rospy.get_param("~keypoint_classifier_model")

        # Set params for 2D Pose Detection
        self.pose_2d_detect  = rospy.get_param( "~pose_2d_detect",   True )
        self.log_show_flag   = rospy.get_param( "~pose_2d_log_show", True )
        self.img_show_flag   = rospy.get_param( "~pose_2d_img_show", True )
        self.img_pub_flag    = rospy.get_param( "~pose_2d_img_pub",  True )

        self.sub_img_topic_name = rospy.get_param("~sub_img_topic_name")
        self.pub_img_topic_name = rospy.get_param("~pub_img_topic_name")
        self.pub_ges_topic_name = rospy.get_param("~pub_ges_topic_name")
        self.pub_hand_lm_topic_name = rospy.get_param("~pub_hand_lm_topic_name")

        self.sub_img          = rospy.Subscriber(self.sub_img_topic_name, Image, self.img_cb)
        self.pub_result_img   = rospy.Publisher(self.pub_img_topic_name, Image, queue_size=10)
        self.pub_gesture      = rospy.Publisher(self.pub_ges_topic_name, String, queue_size=10)
        self.pub_result_array = rospy.Publisher(self.pub_hand_lm_topic_name, KeyPoint2DArray, queue_size=10)
        # self.server           = rospy.Service("~run_ctr", RunCtrl, self.run_ctrl_server)

        # Camera model for pixel->meter conversion
        self.camera_model = None
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb)

        # Publisher for cartesian detections (topic configurable desde launch)
        self.cartesian_pub_topic = rospy.get_param("~cartesian_pub_topic", "/camera/cartesian_detections")
        self.cartesian_pub = rospy.Publisher(self.cartesian_pub_topic, KeyPoint2DArray, queue_size=10)

        # Create a gesture recognition object that loads labels and train model
        self.gesture_detector = GestureRecognition(self.keypoint_classifier_label,
                                                   self.keypoint_classifier_model)

        self.bridge = CvBridge()
        self.cv_fps_calc = CvFpsCalc(buffer_len=10)

    # # RunCtrl Server
    # def run_ctrl_server(self, msg):
    #     self.pose_2d_detect = True if msg.request else False
    #     return RunCtrlResponse(True)

    def camera_info_cb(self, camera_info_msg):
        if self.camera_model is None:
            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(camera_info_msg)
            rospy.loginfo("Camera model initialized for pixel to meter conversion.")

    def pixel_to_cartesian(self, u, v, image_width=None):
        # Invierte la coordenada x (u) antes de calcular la dirección cartesiana
        if image_width is not None:
            u = image_width - u
        if self.camera_model is None:
            return 0.0, 0.0, 0.0
        ray = self.camera_model.projectPixelTo3dRay((u, v))
        return ray[0], ray[1], ray[2]

    def img_cb(self, image_msg):
        """A callback function for the image subscriber

        Args:
            image_msg (sensor_msgs.msg): image message
        """

        if not self.pose_2d_detect:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            image_width = cv_image.shape[1]
            debug_image, gestures, hand_lms, hand_poss, confs = self.gesture_detector.recognize(cv_image)
            # self.pub_gesture.publish(gestures)

            if hand_lms is not None:
                keypoint_array_msg = KeyPoint2DArray()
                cartesian_array_msg = KeyPoint2DArray()
                for hand_lm, hand_pos, gesture, conf in zip(hand_lms, hand_poss, gestures, confs):
                    keypoint_msg = KeyPoint2D()
                    cartesian_msg = KeyPoint2D()

                    keypoint_msg.name       = hand_pos
                    keypoint_msg.gesture    = gesture
                    keypoint_msg.confidence = conf

                    cartesian_msg.name       = hand_pos
                    cartesian_msg.gesture    = gesture
                    cartesian_msg.confidence = conf

                    # Fill pixel keypoints (u,v)
                    keypoint_msg.wrist             = Point(hand_lm[0][0], hand_lm[0][1], 0)
                    keypoint_msg.thumb_cmc         = Point(hand_lm[1][0], hand_lm[1][1], 0)
                    keypoint_msg.thumb_mcp         = Point(hand_lm[2][0], hand_lm[2][1], 0)
                    keypoint_msg.thumb_ip          = Point(hand_lm[3][0], hand_lm[3][1], 0)
                    keypoint_msg.thumb_tip         = Point(hand_lm[4][0], hand_lm[4][1], 0)
                    keypoint_msg.index_finger_mcp  = Point(hand_lm[5][0], hand_lm[5][1], 0)
                    keypoint_msg.index_finger_pip  = Point(hand_lm[6][0], hand_lm[6][1], 0)
                    keypoint_msg.index_finger_dip  = Point(hand_lm[7][0], hand_lm[7][1], 0)
                    keypoint_msg.index_finger_tip  = Point(hand_lm[8][0], hand_lm[8][1], 0)
                    keypoint_msg.middle_finger_mcp = Point(hand_lm[9][0], hand_lm[9][1], 0)
                    keypoint_msg.middle_finger_pip = Point(hand_lm[10][0], hand_lm[10][1], 0)
                    keypoint_msg.middle_finger_dip = Point(hand_lm[11][0], hand_lm[11][1], 0)
                    keypoint_msg.middle_finger_tip = Point(hand_lm[12][0], hand_lm[12][1], 0)
                    keypoint_msg.ring_finger_mcp   = Point(hand_lm[13][0], hand_lm[13][1], 0)
                    keypoint_msg.ring_finger_pip   = Point(hand_lm[14][0], hand_lm[14][1], 0)
                    keypoint_msg.ring_finger_dip   = Point(hand_lm[15][0], hand_lm[15][1], 0)
                    keypoint_msg.ring_finger_tip   = Point(hand_lm[16][0], hand_lm[16][1], 0)
                    keypoint_msg.pinky_mcp         = Point(hand_lm[17][0], hand_lm[17][1], 0)
                    keypoint_msg.pinky_pip         = Point(hand_lm[18][0], hand_lm[18][1], 0)
                    keypoint_msg.pinky_dip         = Point(hand_lm[19][0], hand_lm[19][1], 0)
                    keypoint_msg.pinky_tip         = Point(hand_lm[20][0], hand_lm[20][1], 0)

                    # Fill cartesian keypoints (meters, direction vector) with inverted u
                    for idx, attr in enumerate([
                        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
                        "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
                        "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
                        "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
                        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
                    ]):
                        u = hand_lm[idx][0]
                        v = hand_lm[idx][1]
                        x, y, z = self.pixel_to_cartesian(u, v, image_width=image_width)
                        setattr(cartesian_msg, attr, Point(x, y, z))

                    keypoint_array_msg.header = image_msg.header
                    keypoint_array_msg.data.append(keypoint_msg)

                    cartesian_array_msg.header = image_msg.header
                    cartesian_array_msg.data.append(cartesian_msg)

                # Show the keypoint array if the flag is set to True
                if self.log_show_flag:
                    rospy.loginfo(keypoint_array_msg)
                    rospy.loginfo("Cartesian detections:")
                    rospy.loginfo(cartesian_array_msg)

                # Publish the keypoint array (pixels)
                self.pub_result_array.publish(keypoint_array_msg)
                # Publish the cartesian array (meters/direction)
                self.cartesian_pub.publish(cartesian_array_msg)

            # Show the image if the flag is set to True
            if self.img_show_flag:
                fps = self.cv_fps_calc.get()
                debug_image = self.gesture_detector.draw_fps_info(debug_image, fps)
                cv.imshow('ROS Gesture Recognition', debug_image)
                cv.waitKey(10) # wait for 10 milisecond

            # Publish the image if the flag is set to True
            if self.img_pub_flag:
                try:
                    result_img_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                    result_img_msg.header = image_msg.header
                    self.pub_result_img.publish(result_img_msg)

                except CvBridgeError as error:
                    rospy.logerr(error)

        except CvBridgeError as error:
            rospy.logerr(error)
        

if __name__=="__main__":
    # Initialize the node
    rospy.init_node('hand_sign_recognition_demo', anonymous=True)

    try:
        hand_sign = HandSignRecognition()
        rospy.spin()
    # If we press control + C, the node will stop.
    except rospy.ROSInternalException:
        cv.destroyAllWindows()
        pass
