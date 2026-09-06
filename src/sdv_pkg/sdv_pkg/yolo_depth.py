import rclpy
from rclpy.node import Node
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
            self.listener_callback, 10
        )
        self.depth_raw = self.create_publisher(
            Image,
            'vision/depth_raw',
            10
        )
        self.depth_map = self.create_publisher(
            Image,
            'vision/depth_map',
            10
        )
        self.bridge = CvBridge()
        self.model = YOLO("../../weights/yolo26n-depth.pt")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(source=frame, imgsz=768, verbose=False)
        depth_matrix = results[0].depth.data.cpu().numpy().squeeze()
        depth_matrix = depth_matrix.astype(np.float32)

        raw_msg = self.bridge.cv2_to_imgmsg(depth_matrix, encoding='32FC1')
        raw_msg.header = msg.header
        self.depth_raw.publish(raw_msg)


        norm_depth =cv2.normalize(depth_matrix, None, O, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)

        map_msg = self.bridge.cv2_to_imgmsg(colormap, encoding='bgr8')
        map_msg.header = msg.header
        self.depth_map.publish(map_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

