from launch import LaunchDescription
from launch_ros.actions import Node
import os
def generate_launch_description():
    params = os.path.expanduser('~/amr_gesture_ws/config/recorder.params.yaml')
    return LaunchDescription([
        Node(package='vlm_recorder_pkg', executable='recorder_node',
             name='recorder_node', parameters=[params], output='screen')
    ])
