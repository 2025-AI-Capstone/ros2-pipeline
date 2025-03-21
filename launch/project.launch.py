from launch import LaunchDescription
from launch_ros.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 사용 이미지에 따라 노드 명 변경
    video_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([get_package_share_directory('video_publisher'), 'launch', 'video_publisher.launch.py'])
        ])
    )
    detector_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([get_package_share_directory('detector'), 'launch', 'detector.launch.py'])
        ])
    )
    tracker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([get_package_share_directory('tracker'), 'launch', 'tracker.launch.py'])
        ])
    )
    
    falldetector_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([get_package_share_directory('falldetector'), 'launch', 'falldetector.launch.py'])
        ])
    )

    return LaunchDescription([
        video_launch,
        detector_launch,
        tracker_launch,
        falldetector_launch,
    ])
