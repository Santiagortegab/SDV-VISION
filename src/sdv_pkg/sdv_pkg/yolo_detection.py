import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloDetection(Node):
    def __init__(self):
        super().__init__('yolo_detection')
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.listener_callback,
            10
        )
        self.det_pub = self.create_publisher(
            Detection2DArray,
            'vision/detections',
            10
        )
        self.img_pub = self.create_publisher(
            Image,
            'vision/yolo_annotated',
            10
        )

        self.bridge = CvBridge()
        self.get_logger().info("Cargando modelo YOLO...")
        self.model = YOLO("../../weights/yolo26n.pt")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame)

        det_array = Detection2DArray()
        det_array.header = msg.header

        for box in results[0].boxes:
            det = Detection2D()
            xywh = box.xywh[0].cpu().numpy()
            det.bbox.center.position.x = float(xywh[0])
            det.bbox.center.position.y = float(xywh[1])
            det.bbox.size_x = float(xywh[2])
            det.bbox.size_y = float(xywh[3])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(box.cls[0]))
            hyp.hypothesis.score = float(box.conf[0])

            det.results.append(hyp)
            det_array.detections.append(det)

        self.det_pub.publish(det_array)

        annotaded_frame = results[0].plot()
        out_msg = self.bridge.cv2_to_imgmsg(annotaded_frame, encoding='bgr8')
        out_msg.header = msg.header
        self.img_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


    
