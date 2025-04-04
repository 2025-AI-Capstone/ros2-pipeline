import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stt',
            executable='stt_node',
            name='stt_node',
            output='screen',
            parameters=[
                os.path.join(
                get_package_share_directory('stt'),
                'config',
                'settings.yaml'
                )
            ],
        )
    ])
