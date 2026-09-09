import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
import numpy as np

class BevNode(Node):
    def __init__(self):
        super().__init__('bev_node')
        self.bridge = CvBridge()
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            'vision/bev_markers', 
            10
            )
        self.f_x = 400
        self.f_y = 400
        self.c_x = 320
        self.c_y = 192

        self.class_profiles ={
            '0': ([1.0, 0.0, 0.0], [0.5, 0.5, 1.8]), #Persoa
            '1': ([1.0, 0.5, 0.0], [1.8, 0.6, 1.2]), #Bici
            '2': ([0.0, 0.5, 1.0], [4.5, 2.0, 1.5]), #Coche
        }

        self.sub_det = message_filters.Subscriber(self, Detection2DArray, 'vision/detections')
        self.sub_depth = message_filters.Subscriber(self, Image, 'vision/depth_raw')

        self.ts = message_filters.TimeSynchronizer(
            [self.sub_det, self.sub_depth], 
            10
            )
        self.ts.registerCallback(self.sync_callback)

    def sync_callback(self, det_msg, depth_msg):
        depth_matrix = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        marker_array =MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i, det in enumerate(det_msg.detections):
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            x_min = int(max(cx - w/2, 0))
            y_min = int(max(cy - h/2, 0))
            x_max = int(min(cx + w/2, depth_matrix.shape[1]-1))
            y_max = int(min(cy + h/2, depth_matrix.shape[0]-1))

            depth = depth_matrix[y_min:y_max, x_min:x_max]
            if depth.size == 0: continue
            Z = float(np.max(depth))

            x_cam = ((cx - self.c_x) * Z) / self.f_x
            y_cam = ((cy - self.c_y) * Z) / self.f_y

            marker = Marker()
            marker.header.frame_id = 'camera_link'
            marker.header.stamp = det_msg.header.stamp
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = Z
            marker.pose.position.y = -x_cam
            marker.pose.position.z = -y_cam

            clase_id = det.results[0].hypothesis.class_id
            perfil = self.class_profiles.get(clase_id, ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]))

            marker.color.r, marker.color.g, marker.color.b = perfil[0]
            marker.color.a = 1.0
            
            marker.scale.x = perfil[1][0]
            marker.scale.y = perfil[1][1]
            marker.scale.z = perfil[1][2]

            marker.lifetime = rclpy.duration.Duration(seconds=5).to_msg()

            marker_array.markers.append(marker)

        main_car = Marker()
        main_car.header.frame_id = 'camera_link'
        main_car.header.stamp = det_msg.header.stamp
        main_car.ns = 'main_car'
        main_car.id = 9999
        main_car.type = Marker.CUBE
        main_car.action = Marker.ADD

        main_car.scale.x = 4.0
        main_car.scale.y = 2.0
        main_car.scale.z = 1.0

        main_car.pose.position.x = -1.0
        main_car.pose.position.y = 0.0
        main_car.pose.position.z = -1.2

        main_car.color.r = 0.0
        main_car.color.g = 0.0
        main_car.color.b = 0.0
        main_car.color.a = 1.0

        main_car.lifetime = rclpy.duration.Duration(seconds=5).to_msg()
        marker_array.markers.append(main_car)
        
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = BevNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()