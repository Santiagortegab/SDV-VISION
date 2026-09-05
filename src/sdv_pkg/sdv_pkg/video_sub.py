import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoSubscriber(Node):
    def __init__(self):
        super().__init__('video_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
    def listener_callback(self,msg):
        self.get_logger().info('Recibiendo frame de dashcam...')
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame =cv2.resize(frame, (800,600))
        cv2.imshow('Video SDV', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VideoSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
