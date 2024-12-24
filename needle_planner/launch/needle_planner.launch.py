#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import PythonExpression
from ament_index_python.packages import get_package_share_directory

import os
import yaml

def load_yaml(yaml_file_path):
    """Load YAML configuration file."""
    try:
        with open(yaml_file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load yaml file {yaml_file_path}: {str(e)}")

def load_planner_config(context, planner_type):
    """Load planner-specific configuration."""
    pkg_share = get_package_share_directory('needle_planner')

    # Load default parameters
    default_params_path = os.path.join(pkg_share, 'config', 'default_params.yaml')
    params = load_yaml(default_params_path)

    # Load planner-specific configuration if available
    planner_config_path = os.path.join(pkg_share, 'config', 'planner_configs',
                                      f'{planner_type}.yaml')
    if os.path.exists(planner_config_path):
        planner_params = load_yaml(planner_config_path)
        # Merge configurations (planner-specific takes precedence)
        params.update(planner_params)

    return params

def generate_launch_description():
    """Generate launch description for needle planning system."""
    pkg_share = get_package_share_directory('needle_planner')

    # Declare launch arguments
    planner_type_arg = DeclareLaunchArgument(
        'planner_type',
        default_value='linear_regression',
        description='Type of planner to use: linear_regression, neural_network, geometric, hybrid, or sampling'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug logging'
    )

    # Load parameters based on selected planner
    def launch_setup(context):
        planner_type = LaunchConfiguration('planner_type').perform(context)
        debug = LaunchConfiguration('debug').perform(context)

        # Load configuration
        try:
            params = load_planner_config(context, planner_type)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
            return []

        # Create the planning node
        planning_node = Node(
            package='needle_planner',
            executable='planner_node',
            name='needle_planning_node',
            output='screen',
            parameters=[params],
            arguments=['--ros-args', '--log-level',
                      'debug' if debug.lower() == 'true' else 'info']
        )

        return [planning_node]

    return LaunchDescription([
        planner_type_arg,
        debug_arg,
        OpaqueFunction(function=launch_setup)
    ])

if __name__ == '__main__':
    generate_launch_description()