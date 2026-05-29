from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='central_db_pkg', executable='central_db_node', name='central_db_node', output='screen'),
    ])
