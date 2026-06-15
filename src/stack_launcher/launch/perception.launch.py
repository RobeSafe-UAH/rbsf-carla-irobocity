import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory('carla_agent'), 'urdf', 'car.urdf'
    )
    with open(urdf_path) as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'traffic',
            default_value='50',
            description='Number of traffic vehicles to spawn',
        ),
        Node(
            package='carla_agent',
            executable='agent',
            output='screen',
            arguments=['--traffic', LaunchConfiguration('traffic')],
        ),
        Node(
            package='carla_perception',
            executable='perception',
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', '/workspace/rviz_cfg.rviz'],
        ),
    ])
