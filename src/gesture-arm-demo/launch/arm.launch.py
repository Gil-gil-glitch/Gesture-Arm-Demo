import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(get_package_share_directory('arm_driver'), 'config', 'arm.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        Node(
            package='arm_driver',
            executable='arm_driver',
            name='arm_driver',
            output='screen',
            parameters=[LaunchConfiguration('config'),
                        {'port': LaunchConfiguration('port')}],
        ),
    ])