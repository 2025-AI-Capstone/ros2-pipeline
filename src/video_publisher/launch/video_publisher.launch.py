import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='video_publisher',
            executable='video_publisher_node',
            name='video_publisher_node',
            output='screen',
            parameters=[
                os.path.join(
                get_package_share_directory('video_publisher'),
                'config',
                'settings.yaml'
                )
            ],
        )
    ])
