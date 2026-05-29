from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # UI kiosk node
    import os
    from ament_index_python.packages import get_package_share_directory

    config = os.path.join(
        get_package_share_directory('ui_kiosk_pkg'),
        'config',
        'kiosk_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='ui_kiosk_pkg',
            executable='ui_kiosk_node',
            name='ui_kiosk_node',
            parameters=[config]
        ),
    ])
