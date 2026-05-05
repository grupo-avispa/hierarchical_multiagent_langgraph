#!/usr/bin/env python3

"""
Launches a langgraph agent as an ros pub/sub node with default parameters.

Loads environment variables from a .env file to enable LangSmith tracing.
"""
from dotenv import load_dotenv
import os
import sys

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

# Loading packages from the current virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    site_packages = os.path.join(
        venv_path,
        'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages'
    )
    sys.path.insert(0, site_packages)


def generate_launch_description():
    # Get config .env file
    hierarchical_multiagent_langgraph_dir = get_package_share_directory(
        'hierarchical_multiagent_langgraph')
    dotenv_path = os.path.join(hierarchical_multiagent_langgraph_dir, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    # Get python from current virtual environment
    venv_python = os.path.join(os.environ.get('VIRTUAL_ENV', '/usr'), 'bin', 'python3')

    # Getting directories and launch-files
    default_params_file = os.path.join(
        hierarchical_multiagent_langgraph_dir, 'params', 'default_params.yaml')
    default_mcp_servers_file = os.path.join(
        hierarchical_multiagent_langgraph_dir, 'params', 'sup_mcp.json')
    default_spa_mcp_servers_file = os.path.join(
        hierarchical_multiagent_langgraph_dir, 'params', 'spa_mcp.json')
    default_supervisor_sys_prompt_file = os.path.join(
        hierarchical_multiagent_langgraph_dir, 'templates', 'supervisor_system_prompt.jinja')
    default_agent_sys_prompt_file = os.path.join(
        hierarchical_multiagent_langgraph_dir, 'templates', 'agent_system_prompt.jinja')
    default_template = os.path.join(
        hierarchical_multiagent_langgraph_dir, 'templates', 'qwen3.jinja')
    default_spa_template = os.path.join(
        hierarchical_multiagent_langgraph_dir, 'templates', 'qwen3.jinja')

    # Input parameters declaration
    params_file = LaunchConfiguration('params_file')
    mcp_servers_file = LaunchConfiguration('mcp_servers_file')
    spa_mcp_servers_file = LaunchConfiguration('spa_mcp_servers_file')
    supervisor_sys_prompt_file = LaunchConfiguration('supervisor_sys_prompt_file')
    agent_sys_prompt_file = LaunchConfiguration('agent_sys_prompt_file')
    template_file = LaunchConfiguration('template_file')
    spa_template_file = LaunchConfiguration('spa_template_file')
    log_level = LaunchConfiguration('log-level')

    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS2 parameters file with configuration'
    )
    declare_template_file_arg = DeclareLaunchArgument(
        'template_file',
        default_value=default_template,
        description='Full path to the template file'
    )

    declare_spa_template_file_arg = DeclareLaunchArgument(
        'spa_template_file',
        default_value=default_spa_template,
        description='Full path to the SPA template file'
    )

    declare_mcp_servers_file_arg = DeclareLaunchArgument(
        'mcp_servers_file',
        default_value=default_mcp_servers_file,
        description='Full path to the supervisor MCP servers configuration file'
    )

    declare_spa_mcp_servers_file_arg = DeclareLaunchArgument(
        'spa_mcp_servers_file',
        default_value=default_spa_mcp_servers_file,
        description='Full path to the agent MCP servers configuration file'
    )

    declare_supervisor_sys_prompt_file_arg = DeclareLaunchArgument(
        'supervisor_sys_prompt_file',
        default_value=default_supervisor_sys_prompt_file,
        description='Full path to the supervisor system prompt template file'
    )

    declare_agent_sys_prompt_file_arg = DeclareLaunchArgument(
        'agent_sys_prompt_file',
        default_value=default_agent_sys_prompt_file,
        description='Full path to the agent system prompt template file'
    )

    declare_log_level_arg = DeclareLaunchArgument(
        name='log-level',
        default_value='info',
        description='Logging level (info, debug, ...)'
    )

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'mcp_servers': mcp_servers_file,
        'spa_mcp_servers': spa_mcp_servers_file,
        'system_prompt_file': supervisor_sys_prompt_file,
        'spa_system_prompt_file': agent_sys_prompt_file,
        'template_file': template_file,
        'spa_template_file': spa_template_file,
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True
    )

    # Prepare the langgraph agent node
    langgraph_agent_node = Node(
        package='hierarchical_multiagent_langgraph',
        executable='node',
        name='langgraph_agent_node',
        output='screen',
        prefix=[venv_python, ' -u '],
        parameters=[configured_params],
        arguments=['--ros-args', '--log-level', log_level],
        emulate_tty=True
    )

    return LaunchDescription([
        declare_params_file_arg,
        declare_mcp_servers_file_arg,
        declare_spa_mcp_servers_file_arg,
        declare_template_file_arg,
        declare_spa_template_file_arg,
        declare_supervisor_sys_prompt_file_arg,
        declare_agent_sys_prompt_file_arg,
        declare_log_level_arg,
        langgraph_agent_node,
    ])
