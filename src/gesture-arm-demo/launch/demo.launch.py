"""
Start the whole demo.

  ros2 launch gesture_arm_demo demo.launch.py                 # gesture mode (default)
  ros2 launch gesture_arm_demo demo.launch.py mode:=mimic     # robot copies your right arm
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(get_package_share_directory('gesture-arm-demo'), 'config', 'arm.yaml')
    mode = LaunchConfiguration('mode')
    is_gesture = IfCondition(PythonExpression(["'", mode, "' == 'gesture'"]))
    is_mimic = IfCondition(PythonExpression(["'", mode, "' == 'mimic'"]))

    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='gesture', choices=['gesture', 'mimic']),
        DeclareLaunchArgument('config', default_value=default_config),
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('camera', default_value='0'),

        Node(
            package='gesture-arm-demo',
            executable='driver_node',
            name='arm_driver',
            output='screen',
            parameters=[LaunchConfiguration('config'),
                        {'port': LaunchConfiguration('port')}],
        ),

        # ---- mode:=gesture ----
        Node(
            package='gesture-arm-demo',
            executable='gesture_detector',
            name='gesture_detector',
            output='screen',
            parameters=[{'camera_index': LaunchConfiguration('camera')}],
            condition=is_gesture,
        ),
        Node(
            package='gesture-arm-demo',
            executable='gesture_to_motion',
            name='gesture_to_motion',
            output='screen',
            condition=is_gesture,
        ),

        # ---- mode:=mimic ----
        Node(
            package='gesture-arm-demo',
            executable='pose_mimic',
            name='pose_mimic',
            output='screen',
            parameters=[{'camera_index': LaunchConfiguration('camera')}],
            condition=is_mimic,
        ),
    ])