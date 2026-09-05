from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
    

        Node(
            package='sdv_pkg',
            executable='video_pub',
            name='publicador_video',
            output='screen'
        ),
        Node(
            package='sdv_pkg',
            executable='video_sub',
            name='suscriptor_video',
            output='screen'
        )
    ])