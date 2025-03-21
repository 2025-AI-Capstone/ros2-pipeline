import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_publisher',
            executable='image_publisher_node',
            name='image_publisher_node',
            output='screen',
            parameters=[
                os.path.join(
                get_package_share_directory('image_publisher'),
                'config',
                'settings.yaml'
                )
            ],
        )
    ])
