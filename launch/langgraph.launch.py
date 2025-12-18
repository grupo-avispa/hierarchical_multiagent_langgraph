#!/usr/bin/env python3

"""
Launches a langgraph agent as an ros pub/sub node with default parameters.

Loads environment variables from a .env file to enable LangSmith tracing.
"""
import os
import sys

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
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

from dotenv import load_dotenv

def generate_launch_description():
    # Get config .env file
    hierarchical_multiagent_langgraph_dir = get_package_share_directory('hierarchical_multiagent_langgraph')
    dotenv_path = os.path.join(hierarchical_multiagent_langgraph_dir, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    # Get python from current virtual environment
    venv_python = os.path.join(os.environ.get('VIRTUAL_ENV', '/usr'), 'bin', 'python3')
    
    # Getting directories and launch-files
    default_params_file = os.path.join(hierarchical_multiagent_langgraph_dir, 'params', 'default_params.yaml')
    # default_mcp_servers_file = os.path.join(hierarchical_multiagent_langgraph_dir, 'params', 'langgraph_mcp.json')
    # default_sys_prompt_file = os.path.join(hierarchical_multiagent_langgraph_dir, 'templates', 'system_prompt.jinja')
    # default_chat_template_file = os.path.join(hierarchical_multiagent_langgraph_dir, 'templates', 'qwen3.jinja')

    # Input parameters declaration
    params_file = LaunchConfiguration('params_file')
    # mcp_servers_file = LaunchConfiguration('mcp_servers_file')
    # sys_prompt_file = LaunchConfiguration('sys_prompt_file')
    # chat_template_file = LaunchConfiguration('chat_template_file')
    log_level = LaunchConfiguration('log-level')

    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS2 parameters file with configuration'
    )

    # declare_mcp_servers_file_arg = DeclareLaunchArgument(
    #     'mcp_servers_file',
    #     default_value=default_mcp_servers_file,
    #     description='Full path to the MCP servers configuration file'
    # )

    # declare_sys_prompt_file_arg = DeclareLaunchArgument(
    #     'sys_prompt_file',
    #     default_value=default_sys_prompt_file,
    #     description='Full path to the system prompt template file'
    # )

    # declare_chat_template_file_arg = DeclareLaunchArgument(
    #     'chat_template_file',
    #     default_value=default_chat_template_file,
    #     description='Full path to the chat template file'
    # )

    declare_log_level_arg = DeclareLaunchArgument(
        name='log-level',
        default_value='info',
        description='Logging level (info, debug, ...)'
    )

    # # Create our own temporary YAML files that include substitutions
    # param_substitutions = {
    #     'mcp_servers': mcp_servers_file,
    #     'system_prompt_file': sys_prompt_file,
    #     'model_chat_template_file': chat_template_file,
    # }

    # configured_params = RewrittenYaml(
    #     source_file=params_file,
    #     root_key='',
    #     param_rewrites=param_substitutions,
    #     convert_types=True
    # )

    # Prepare the langgraph agent node
    langgraph_agent_node = Node(
        package='hierarchical_multiagent_langgraph',
        executable='node',
        name='langgraph_agent_node',
        output='screen',
        prefix=[venv_python, ' -u '],
        parameters=[params_file],
        arguments=['--ros-args', '--log-level', log_level],
    )

    return LaunchDescription([
        declare_params_file_arg,
        # declare_mcp_servers_file_arg,
        # declare_sys_prompt_file_arg,
        # declare_chat_template_file_arg,
        declare_log_level_arg,
        langgraph_agent_node,
    ])