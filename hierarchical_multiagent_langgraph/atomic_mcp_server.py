#!/usr/bin/env python3

# Copyright (c) 2025 Jose Galeas
# Copyright (c) 2025 Grupo Avispa, DTE, Universidad de Málaga
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.action import ActionClient
from object_with_region.msg import ObjectRegion3DArray
from audio_common_msgs.action import TTS
from llm_interactions_msgs.srv import RetrieveDocuments
from semantic_navigation_msgs.srv import GenerateRandomGoals
from nav2_msgs.action import NavigateThroughPoses, DockRobot
import threading
import time
from fastmcp import FastMCP
import json
from datetime import date
import requests


# Initialize MCP server
mcp = FastMCP("atomic_mcp_server")

# Global variable for ROS node
ros_node = None


class GetObjectRegionNode(Node):
    """ROS2 node to get the region of a detected object."""

    def __init__(self):
        super().__init__('get_object_region_mcp')
        
        # Configurable parameters
        self.declare_parameter('objects_topic', '/object_detection/objects_with_region')
        self.declare_parameter('objects_timeout', 2.5)  # seconds
        
        self.objects_topic = self.get_parameter('objects_topic').value
        self.objects_timeout = self.get_parameter('objects_timeout').value
        
        # Store received objects
        self.objects = None
        self.objects_lock = threading.Lock()
        
        # Configure QoS for sensor data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscriber
        self.objects_sub = self.create_subscription(
            ObjectRegion3DArray,
            self.objects_topic,
            self.objects_callback,
            qos_profile
        )
        
        # Create TTS action client
        self._tts_action_client = ActionClient(self, TTS, '/say')
        
        # Create RAG service client
        self._rag_client = self.create_client(RetrieveDocuments, '/retrieve_documents')
        
        # Create navigation clients
        self._generate_random_goals_client = self.create_client(
            GenerateRandomGoals, '/generate_random_goals')
        self._navigate_action_client = ActionClient(
            self, NavigateThroughPoses, '/navigate_through_poses')
        
        # Create docking action client
        self._dock_action_client = ActionClient(self, DockRobot, '/dock_robot')
        
        self.get_logger().info(f'Subscribed to objects topic: {self.objects_topic}')
    
    def objects_callback(self, msg):
        """Callback to update detected objects."""
        with self.objects_lock:
            self.objects = msg
            self.get_logger().info(
                f'Received objects data with {len(msg.objects)} objects'
            )
    
    def get_valid_objects_data(self):
        """
        Get valid objects data with timeout handling.
        
        Returns:
            ObjectRegion3DArray or None if no valid data available
        """
        with self.objects_lock:
            current_objects = self.objects
        
        # If we haven't received data, wait until timeout
        if current_objects is None:
            self.get_logger().info(
                f'Waiting for objects data (timeout: {self.objects_timeout} seconds)...'
            )
            start_time = time.time()
            rate = self.create_rate(10.0)  # 10 Hz
            
            while time.time() - start_time < self.objects_timeout:
                with self.objects_lock:
                    if self.objects is not None:
                        current_objects = self.objects
                        break
                rate.sleep()
            
            if current_objects is None:
                self.get_logger().warn(
                    f'No objects data received within timeout period '
                    f'({self.objects_timeout} seconds)'
                )
                return None
        
        # Validate that data is recent enough
        msg_time = Time.from_msg(current_objects.header.stamp)
        current_time = self.get_clock().now()
        age_seconds = (current_time - msg_time).nanoseconds / 1e9
        
        if age_seconds > self.objects_timeout:
            self.get_logger().warn(
                f'Objects data has expired (age: {age_seconds:.1f} seconds, '
                f'timeout: {self.objects_timeout} seconds)'
            )
            return None
        
        return current_objects
    
    def find_object_region(self, object_name: str):
        """
        Search for an object by name and get its region.
        
        Args:
            object_name: Object name to search for (class_id)
            
        Returns:
            dict with 'found', 'region' and message
        """
        # Get valid objects data
        current_objects = self.get_valid_objects_data()
        
        if current_objects is None:
            return {
                'found': False,
                'region': None,
                'message': 'Could not get valid objects data'
            }
        
        self.get_logger().info(f'Searching for object: {object_name}')
        
        # Search for the object in current data
        for obj_region in current_objects.objects:
            if len(obj_region.object.results) > 0:
                class_id = obj_region.object.results[0].hypothesis.class_id
                if class_id == object_name:
                    self.get_logger().info(
                        f'Object found with class_id: {class_id} in region: {obj_region.region}'
                    )
                    return {
                        'found': True,
                        'region': obj_region.region,
                        'message': f'Object "{object_name}" found in region "{obj_region.region}"'
                    }
        
        self.get_logger().info(
            f'Object "{object_name}" not found in current objects data'
        )
        return {
            'found': False,
            'region': None,
            'message': f'Object "{object_name}" not found'
        }
    
    def send_tts(self, text: str, timeout: float = 10.0):
        """
        Send text to TTS action server.
        
        Args:
            text: Text to speak
            timeout: Timeout in seconds to wait for action server
            
        Returns:
            dict with 'success' and 'message'
        """
        # Wait for action server
        if not self._tts_action_client.wait_for_server(timeout_sec=timeout):
            return {
                'success': False,
                'message': f'TTS action server not available after {timeout} seconds'
            }
        
        # Create goal
        goal_msg = TTS.Goal()
        goal_msg.text = text
        
        # Send goal
        self.get_logger().info(f'Sending TTS goal: "{text}"')
        send_goal_future = self._tts_action_client.send_goal(goal_msg)
        
        if send_goal_future is None:
            return {
                'success': False,
                'message': 'TTS goal was failed'
            }
        
        self.get_logger().info('TTS goal accepted, waiting for result...')
        
        return {
            'success': True,
            'message': f'Successfully spoke: "{text}"'
        }
    
    def call_rag(self, query: str, k: int = 3, filters: str = "", enable_refinement: bool = False, timeout: float = 30.0):
        """
        Call RAG service to retrieve documents.
        
        Args:
            query: Search query string
            k: Number of documents to retrieve (default: 3)
            filters: Optional filters for search
            enable_refinement: Enable RAG refinement (default: False)
            timeout: Timeout in seconds to wait for service
            
        Returns:
            dict with 'success', 'status', 'total_results', 'documents', and optionally 'refinement'
        """
        # Wait for service
        if not self._rag_client.wait_for_service(timeout_sec=timeout):
            return {
                'success': False,
                'status': 'error',
                'message': f'RAG service not available after {timeout} seconds',
                'total_results': 0,
                'documents': []
            }
        
        # Create request
        request = RetrieveDocuments.Request()
        request.query = query
        request.k = k
        request.filters = filters
        request.refine_rag = enable_refinement
        
        # Call service
        self.get_logger().info(f'Calling RAG service with query: "{query}"')
        future = self._rag_client.call(request)
        print(future)
        
        if future is None:
            return {
                'success': False,
                'status': 'timeout',
                'message': 'RAG service call timed out',
                'total_results': 0,
                'documents': []
            }
        
        # Process response
        documents = []
        for doc in future.results:
            documents.append({
                'content': doc.content,
                'score': doc.score if hasattr(doc, 'score') else 0.0
            })
        
        result = {
            'success': True,
            'status': future.status,
            'total_results': future.total_results,
            'documents': documents
        }
        
        # Add refinement if enabled and available
        if enable_refinement and future.results:
            result['refinement'] = future.results[0].content
        
        self.get_logger().info(f'RAG service returned {future.total_results} results')
        return result
    
    def move_to_region(self, region_name: str, number_of_goals: int = 1, timeout: float = 60.0):
        """
        Move to a specific region by first generating a goal and then navigating to it.
        
        Args:
            region_name: Name of the region to navigate to
            number_of_goals: Number of goals to generate (default: 1)
            timeout: Timeout in seconds for the operation
            
        Returns:
            dict with 'success', 'message', and navigation details
        """
        # Step 1: Generate random goals in the region
        if not self._generate_random_goals_client.wait_for_service(timeout_sec=5.0):
            return {
                'success': False,
                'message': 'Generate random goals service not available'
            }
        
        # Create request for generating goals
        goal_request = GenerateRandomGoals.Request()
        goal_request.region_name = region_name
        goal_request.n = number_of_goals
        goal_request.yaw = 0.0
        goal_request.orientation = 0.0  # INSIDE orientation
        goal_request.border = 0.0
        
        self.get_logger().info(f'Generating goals for region: {region_name}')
        goal_response = self._generate_random_goals_client.call(goal_request)
        
        if goal_response is None:
            return {
                'success': False,
                'message': 'Goal generation timed out'
            }
        
        # Check if goals were generated
        if not goal_response.goals or len(goal_response.goals.poses) == 0:
            return {
                'success': False,
                'message': f'No goals generated for region "{region_name}"'
            }
        
        goals_count = len(goal_response.goals.poses)
        self.get_logger().info(f'Generated {goals_count} goals for region {region_name}')
        
        # Step 2: Navigate to the generated goals
        if not self._navigate_action_client.wait_for_server(timeout_sec=10.0):
            return {
                'success': False,
                'message': 'Navigation action server not available',
                'goals_generated': goals_count
            }
        
        # Create navigation goal
        nav_goal = NavigateThroughPoses.Goal()
        nav_goal.poses = goal_response.goals.poses
        nav_goal.behavior_tree = ''  # Use default behavior tree
        
        self.get_logger().info(f'Starting navigation to region: {region_name}')
        goal_result = self._navigate_action_client.send_goal(nav_goal)
        
        if goal_result is None:
            return {
                'success': False,
                'message': 'Navigation goal was rejected or robot was not able to navigate',
                'goals_generated': goals_count
            }
        
        
        return {
            'success': True,
            'message': f'Successfully navigated to region "{region_name}"',
            'goals_generated': goals_count
        }
    
    def charge_robot(self, dock_id: str = "wall_dock", dock_type: str = "scitos_dock", navigate_to_staging_pose: bool = True, timeout: float = 120.0):
        """
        Dock the robot to a charging station.
        
        Args:
            dock_id: ID or name of the dock (optional if dock has default)
            dock_type: Type of dock plugin (optional)
            navigate_to_staging_pose: Whether to autonomously navigate to staging pose (default: True)
            timeout: Timeout in seconds for docking operation
            
        Returns:
            dict with 'success', 'message', 'num_retries', and error information
        """
        # Wait for action server
        if not self._dock_action_client.wait_for_server(timeout_sec=10.0):
            return {
                'success': False,
                'message': 'Dock robot action server not available'
            }
        
        # Create docking goal
        dock_goal = DockRobot.Goal()
        
        if dock_id:
            dock_goal.use_dock_id = True
            dock_goal.dock_id = dock_id
        else:
            dock_goal.use_dock_id = False
        
        if dock_type:
            dock_goal.dock_type = dock_type
        
        dock_goal.navigate_to_staging_pose = navigate_to_staging_pose
        dock_goal.max_staging_time = 1000.0  # Default max staging time
        
        self.get_logger().info(f'Starting docking operation (dock_id: {dock_id or "default"})')
        result_goal = self._dock_action_client.send_goal(dock_goal)
        
        if result_goal is None:
            return {
                'success': False,
                'message': 'Docking goal was rejected or robot was not able to dock'
            }
        
        self.get_logger().info('Docking goal accepted, waiting for completion...')
        
        if result_goal.success:
            return {
                'success': True,
                'message': 'Successfully docked to charging station',
                'num_retries': result_goal.num_retries
            }
        else:
            return {
                'success': False,
                'message': f'Docking failed: {result_goal.message}',
                'num_retries': result_goal.num_retries
            }


@mcp.tool(
    name = 'charge_robot',
    description = '''Docks the robot to a charging station for battery charging'''
)
def charge_robot(dock_id: str = "", navigate_to_staging_pose: bool = True) -> dict:
    """
    Commands the robot to dock at a charging station.
    
    Args:
        dock_id: ID or name of the charging dock (leave empty for default dock)
        navigate_to_staging_pose: Whether to autonomously navigate to staging pose (default: True)
        
    Returns:
        dict with 'success', 'message', 'num_retries', and error information if applicable
    """
    
    if ros_node is None:
        return {
            'success': False,
            'message': 'ROS node not initialized'
        }
    
    return ros_node.charge_robot(dock_id=dock_id, navigate_to_staging_pose=navigate_to_staging_pose)


@mcp.tool(
    name = 'move_to_region',
    description = '''Navigates the robot to a specific named region in the environment'''
)
def move_to_region(region_name: str, number_of_goals: int = 1) -> dict:
    """
    Moves the robot to a specific region by generating navigation goals and executing navigation.
    
    Args:
        region_name: Name of the region to navigate to
        number_of_goals: Number of navigation goals to generate (default: 1)
        
    Returns:
        dict with 'success', 'message', and 'goals_generated' count
    """
    
    if ros_node is None:
        return {
            'success': False,
            'message': 'ROS node not initialized'
        }
    
    return ros_node.move_to_region(region_name, number_of_goals=number_of_goals)


@mcp.tool(
    name = 'call_rag',
    description = '''Retrieves relevant documents from the RAG (Retrieval-Augmented Generation) system based on a search query'''
)
def retrieve_documents(query: str, k: int = 3, enable_refinement: bool = False) -> dict:
    """
    Searches for and retrieves relevant documents using the RAG system.
    
    Args:
        query: Search query string
        k: Number of documents to retrieve (default: 3)
        enable_refinement: Enable RAG refinement to get refined answer (default: False)
        
    Returns:
        dict with 'success', 'status', 'total_results', 'documents' list, and optionally 'refinement'
    """
    
    if ros_node is None:
        return {
            'success': False,
            'status': 'error',
            'message': 'ROS node not initialized',
            'total_results': 0,
            'documents': []
        }
    
    return ros_node.call_rag(query, k=k, enable_refinement=enable_refinement)


@mcp.tool(
    name = 'say_text',
    description = '''Speaks the given text using the robot's text-to-speech system'''
)
def say_text(text: str) -> dict:
    """
    Speaks text using the robot's TTS action.
    
    Args:
        text: Text to speak
        
    Returns:
        dict with 'success' (bool) and 'message' (str)
    """
    
    if ros_node is None:
        return {
            'success': False,
            'message': 'ROS node not initialized'
        }
    
    return ros_node.send_tts(text)


@mcp.tool(
    name = 'get_object_region',
    description = '''Gets the region of an object detected by the vision system'''
)
def get_object_region(object_name: str) -> dict:
    """
    Searches for an object by name in the array of detected objects
    from the vision system and returns the region where it is located.
    
    Args:
        object_name: Name of the object to search for (class_id)
        
    Returns:
        dict with 'found' (bool), 'region' (str or None) and 'message' (str)
    """
    
    if ros_node is None:
        return {
            'found': False,
            'region': None,
            'message': 'ROS node not initialized'
        }
    
    return ros_node.find_object_region(object_name)

# ============= WEATHER =============
@mcp.tool(
    name='get_weather',
    description="""Get current weather information for a specified city.
This tool retrieves the current weather conditions for a given city using the Open-Meteo API.
The required tool input argument is the city name as a string.
""",
    tags={'weather', 'current', 'forecast', 'city'}
)
async def get_weather(city: str) -> str:
    if not city:
        error_response = {
            'status': 'error',
            'message': 'Please provide a city name.',
            'results': []
        }
        return json.dumps(error_response, indent=2)

    try:
        # Geocoding API to get latitude and longitude for the city
        result_city = requests.get(url='https://geocoding-api.open-meteo.com/v1/search?name=' + city)
        location = result_city.json()
        lon=str(location['results'][0]['longitude'])
        lat=str(location['results'][0]['latitude'])
        # Get today's date
        today = date.today().isoformat()
        # Open-Meteo endpoint
        url = "https://api.open-meteo.com/v1/forecast"
        # Parameters
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "start_date": today,
            "end_date": today
        }

        # Do the request
        response = requests.get(url, params=params)
        data = response.json()
        # Extract results
        daily = data["daily"]
        tmax = daily["temperature_2m_max"][0]
        tmin = daily["temperature_2m_min"][0]
        precip = daily["precipitation_probability_max"][0]
        weather_data = {
            'temp_max': tmax,
            'temp_min': tmin,
            'precipitation_probability': precip,
        }
        tool_response = {
                'status': 'success',
                'message': 'Weather data retrieved successfully.',
                'city': city,
                'results': weather_data
            }
        return json.dumps(tool_response, indent=2)

    except Exception as e:
        # await ctx.error(f'Error retrieving weather data: {e}')
        error_response = {
            'status': 'error',
            'message': f'Error retrieving weather data: {e}',
            'results': []
        }
        return json.dumps(error_response, indent=2)
      
def ros_spin_thread():
    """Thread to execute ROS spin."""
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass


def main(args=None):
    global ros_node
    
    rclpy.init(args=args)
    ros_node = GetObjectRegionNode()
    
    # Start ROS thread
    ros_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    ros_thread.start()
    
    # Run MCP server with HTTP transport
    try:
        mcp.run(transport="http", host="0.0.0.0", port=8988)
    except KeyboardInterrupt:
        pass
    finally:
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
