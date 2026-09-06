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
        self.f_x = 800
        self.f_y = 800
        self.c_x = 640
        self.c_y = 360

        self.sub_det = message_filters.Subscriber(self, Detection2DArray, 'vision/detections')
        self.sub_depth = message_filters.Subscriber(self, Image, 'vision/depth_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_det, self.sub_depth], queue_size=10, slop=0.05)
        self.ts.registerCallBack(self.sync_callback)

    def sync_callback(self, det_msg, depth_msg):
        depth_matrix = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        marker_array =MarkerArray()

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
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD

            marker.pose.position.x = Z
            marker.pose.position.y = -x_cam
            marker.pose.position.z = -y_cam

            clase_id = det.results[0].hypothesis.class_id
            if clase_id == '0':    # Persona
                marker.mesh_resource = "package://sdv_pkg/meshes/person.stl"
            elif clase_id == '1':  # Bicicleta
                marker.mesh_resource = "package://sdv_pkg/meshes/bicicleta.stl"
            elif clase_id == '2':  # Auto
                marker.mesh_resource = "package://sdv_pkg/meshes/Car.dae"
            else:
                marker.type = Marker.CUBE

            marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0
            marker.color.a = 1.0; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
            marker.mesh_use_embedded_materials = True
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 100000000

            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)