import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_node')
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.listener_callback, 
            qos_profile_sensor_data
        )
        self.depth_raw = self.create_publisher(
            Image,
            'vision/depth_raw',
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info("Cargando modelo depth YOLO...")
        self.model = YOLO("/home/santiagortegab/SDV_VISION_ws/SDV-VISION/weights/yolo26n-depth.engine")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, (640, 384))
        results = self.model.predict(source=frame, imgsz=(384, 640), device=0)
        depth_matrix = results[0].depth.data.cpu().numpy().squeeze()
        depth_matrix = depth_matrix.astype(np.float32)

        raw_msg = self.bridge.cv2_to_imgmsg(depth_matrix, encoding='32FC1')
        raw_msg.header = msg.header
        self.depth_raw.publish(raw_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

