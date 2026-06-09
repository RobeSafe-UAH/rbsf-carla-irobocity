from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', '/workspace/rviz_cfg.rviz'],
        ),
    ])
