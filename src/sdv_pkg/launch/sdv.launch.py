import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    venv_python = '/home/santiagortegab/SDV_VISION_ws/SDV-VISION/envSDV/bin/python3'
    scripts_dir = '/home/santiagortegab/SDV_VISION_ws/SDV-VISION/src/sdv_pkg/sdv_pkg'
    return LaunchDescription([

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['1.0', '0.0', '1.2', '0', '0', '0', 'base_link', 'camera_link']
        ),

        Node(
            package='sdv_pkg',
            executable='video_pub',
            name='publicador_video'
        ),

        ExecuteProcess(
            cmd=[venv_python, os.path.join(scripts_dir, 'yolo_detection.py')],
            name='yolo_detection',
            output='screen'
        ),
        ExecuteProcess(
            cmd=[venv_python, os.path.join(scripts_dir, 'yolo_depth.py')],
            name='yolo_depth',
            output='screen'
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
        )
    ])