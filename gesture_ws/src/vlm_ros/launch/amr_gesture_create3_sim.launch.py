#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

def generate_launch_description():
    # Get package directories
    pkg_create3_sim = get_package_share_directory('create3_sim_integration')
    pkg_vlm_ros = get_package_share_directory('vlm_ros')

    # Path to simulation launch
    sim_launch = PathJoinSubstitution(
        [pkg_create3_sim, 'launch', 'create3_gazebo_sim.launch.py'])

    # Path to gesture pipeline launch
    pipeline_launch = PathJoinSubstitution(
        [pkg_vlm_ros, 'launch', 'pipeline.launch.py'])

    # Declare use_rviz argument
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to start RViz'
    )

    # Include Create3 Simulation
    # This spawns the robot in Gazebo with /AMR namespace
    sim_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([sim_launch]),
        launch_arguments={'use_rviz': LaunchConfiguration('use_rviz')}.items()
    )

    # Include Gesture Pipeline
    # This starts camera, VLM, telemetry, etc.
    # Telemetry node is configured to publish to /AMR/cmd_vel by default
    pipeline_action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pipeline_launch])
    )

    ld = LaunchDescription()
    ld.add_action(use_rviz_arg)
    ld.add_action(sim_action)
    ld.add_action(pipeline_action)

    return ld
