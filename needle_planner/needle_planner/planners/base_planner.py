#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from geometry_msgs.msg import Point, PolygonStamped, Point32
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


class BasePlanner(ABC):
    """
    Abstract base class for needle planners.

    This class defines the interface that all needle planners must implement.
    It provides common functionality and ensures consistent behavior across
    different planning approaches.
    """

    def __init__(self, node: Optional[Node] = None):
        """
        Initialize the base planner.

        Args:
            node (Optional[Node]): ROS2 node instance for logging and parameters
        """
        self.node = node
        self._parameters: Dict[str, Any] = {}

        if self.node:
            self._load_common_parameters()

    def _load_common_parameters(self) -> None:
        """Load common parameters shared by all planners."""
        if not self.node:
            return

        try:
            # Declare and get validation parameters
            self.node.declare_parameter('validation.max_depth', 120.0)
            self.node.declare_parameter('validation.max_lateral', 50.0)
            self.node.declare_parameter('validation.safety_margin', 2.0)
            self.node.declare_parameter('validation.min_step', 0.5)

            self._parameters['max_depth'] = self.node.get_parameter(
                'validation.max_depth').value
            self._parameters['max_lateral'] = self.node.get_parameter(
                'validation.max_lateral').value
            self._parameters['safety_margin'] = self.node.get_parameter(
                'validation.safety_margin').value
            self._parameters['min_step'] = self.node.get_parameter(
                'validation.min_step').value

        except rclpy.exceptions.ParameterException as e:
            self.log_error(f"Failed to load parameters: {str(e)}")
            # Set default values
            self._parameters.update({
                'max_depth': 120.0,
                'max_lateral': 50.0,
                'safety_margin': 2.0,
                'min_step': 0.5
            })

    def log_info(self, msg: str) -> None:
        """Log information message if node is available."""
        if self.node:
            self.node.get_logger().info(msg)

    def log_warn(self, msg: str) -> None:
        """Log warning message if node is available."""
        if self.node:
            self.node.get_logger().warn(msg)

    def log_error(self, msg: str) -> None:
        """Log error message if node is available."""
        if self.node:
            self.node.get_logger().error(msg)

    @abstractmethod
    def compute_plan(self, target: Point) -> PolygonStamped:
        """
        Compute a needle path plan to reach the target point.

        Args:
            target (Point): Target point in 3D space

        Returns:
            PolygonStamped: A series of points representing the planned path

        Raises:
            ValueError: If target point is invalid or unreachable
            RuntimeError: If planning fails
        """
        pass

    def _create_polygon_stamped(self,
                              points: List[Tuple[float, float, float]]
                              ) -> PolygonStamped:
        """
        Convert a list of points to a PolygonStamped message.

        Args:
            points (List[Tuple[float, float, float]]): List of (x,y,z) points

        Returns:
            PolygonStamped: ROS message containing the path points
        """
        msg = PolygonStamped()
        if self.node:
            msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "needle_frame"

        for x, y, z in points:
            point = Point32()
            point.x = float(x)
            point.y = float(y)
            point.z = float(z)
            msg.polygon.points.append(point)

        return msg

    def validate_target(self, target: Point) -> bool:
        """
        Validate that the target point is reachable and valid.

        Args:
            target (Point): Target point to validate

        Returns:
            bool: True if target is valid, False otherwise
        """
        # Check for NaN values
        if any(np.isnan([target.x, target.y, target.z])):
            self.log_error("Target contains NaN values")
            return False

        # Check if target is within workspace limits
        lateral_distance = np.sqrt(target.x**2 + target.y**2)
        if lateral_distance > self._parameters['max_lateral']:
            self.log_error(f"Target lateral distance {lateral_distance:.2f} mm exceeds "
                         f"limit of {self._parameters['max_lateral']:.2f} mm")
            return False

        if target.z > self._parameters['max_depth']:
            self.log_error(f"Target depth {target.z:.2f} mm exceeds "
                         f"limit of {self._parameters['max_depth']:.2f} mm")
            return False

        if target.z < 0:
            self.log_error("Target depth cannot be negative")
            return False

        return True

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value with fallback to default.

        Args:
            name (str): Parameter name
            default (Any): Default value if parameter is not found

        Returns:
            Any: Parameter value
        """
        return self._parameters.get(name, default)

    def validate_path(self,
                     points: List[Tuple[float, float, float]],
                     check_curvature: bool = True) -> bool:
        """
        Validate a complete path.

        Args:
            points (List[Tuple[float, float, float]]): Path points to validate
            check_curvature (bool): Whether to check path curvature

        Returns:
            bool: True if path is valid, False otherwise
        """
        if len(points) < 2:
            self.log_error("Path must contain at least 2 points")
            return False

        # Check point spacing
        for i in range(1, len(points)):
            dist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
            if dist < self._parameters['min_step']:
                self.log_error(f"Point spacing {dist:.2f} mm is below minimum "
                             f"of {self._parameters['min_step']:.2f} mm")
                return False

        if check_curvature and len(points) > 2:
            # Compute and check curvature if required
            from ..utils.path_utils import compute_curvature
            try:
                curvature = compute_curvature(points)
                max_curvature = self.get_parameter('max_curvature', 0.01)
                if np.any(curvature > max_curvature):
                    self.log_error(f"Path curvature exceeds maximum of {max_curvature}")
                    return False
            except Exception as e:
                self.log_error(f"Failed to compute path curvature: {str(e)}")
                return False

        return True