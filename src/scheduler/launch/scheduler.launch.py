import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='scheduler',
            executable='scheduler_node',
            name='scheduler_node',
            output='screen',
            parameters=[
                os.path.join(
                get_package_share_directory('scheduler'),
                'config',
                'settings.yaml'
                )
            ],
        )
    ])
