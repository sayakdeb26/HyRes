from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="vlm_bridge_pkg",
            executable="bridge_node",
            name="bridge_node",
            output="screen",
            parameters=[{
                "confirm_timeout_s": 20.0,
                "wait_clip_timeout_s": 5.0,
                # Not used by the service client directly, but fine to keep if you reference it.
            }],
        )
    ])
