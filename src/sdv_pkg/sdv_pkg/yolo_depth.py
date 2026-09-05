import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

def DepthNode (Node):
    def __init__(self):
        super().__init__('depth_node')
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.listener_callback, 10
        )
        self.publisher_ = self.create_publisher(
            Image,
            'vision/depth_map',
            10
        )
        self.bridge = CvBridge()
        self.model = YOLO("../../weights/yolo26n-depth.pt")
    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        depth_map = model.predict(source=frame, imgsz=768, verbose=False)
        gray = cv2.cvtColorMap(depth_map, cv2.COLORMAP_INFERNO)
        depth_colormap = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        out_msg = self.bridge.cv2_to_imgmsg(depth_colormap, encoding='bgr8')
        out_msg.header = msg.header
        self.publisher_.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

