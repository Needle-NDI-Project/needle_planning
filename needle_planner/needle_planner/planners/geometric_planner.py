#!/usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Point, PolygonStamped
from typing import List, Tuple, Optional

from .base_planner import BasePlanner


class GeometricPlanner(BasePlanner):
    """
    Geometric-based needle path planner.

    This planner uses geometric principles to generate needle paths.
    It assumes the needle follows a circular arc in 3D space, which is a
    reasonable approximation for many needle insertion scenarios.
    """

    def __init__(self, node=None):
        """
        Initialize the geometric planner.

        Args:
            node (Optional[Node]): ROS2 node for logging and parameters
        """
        super().__init__(node)
        self._load_parameters()

    def _load_parameters(self) -> None:
        """Load planner-specific parameters."""
        if not self.node:
            # Set defaults if no node is provided
            self._parameters.update({
                'max_curvature': 0.01,  # 1/mm
                'num_points': 50,
                'min_step': 0.5,  # mm
                'max_angle': 45.0,  # degrees
                'smoothing_enabled': True,
                'smoothing_window': 3
            })
            return

        try:
            param_prefix = 'geometric.planning.'

            # Planning parameters
            self.node.declare_parameter(f'{param_prefix}max_curvature', 0.01)
            self.node.declare_parameter(f'{param_prefix}num_points', 50)
            self.node.declare_parameter(f'{param_prefix}min_step', 0.5)
            self.node.declare_parameter(f'{param_prefix}max_angle', 45.0)

            # Smoothing parameters
            self.node.declare_parameter('geometric.smoothing.enabled', True)
            self.node.declare_parameter('geometric.smoothing.window_size', 3)

            # Update parameters
            self._parameters.update({
                'max_curvature': self.node.get_parameter(
                    f'{param_prefix}max_curvature').value,
                'num_points': self.node.get_parameter(
                    f'{param_prefix}num_points').value,
                'min_step': self.node.get_parameter(
                    f'{param_prefix}min_step').value,
                'max_angle': self.node.get_parameter(
                    f'{param_prefix}max_angle').value,
                'smoothing_enabled': self.node.get_parameter(
                    'geometric.smoothing.enabled').value,
                'smoothing_window': self.node.get_parameter(
                    'geometric.smoothing.window_size').value
            })

        except Exception as e:
            self.log_error(f"Failed to load parameters: {str(e)}")
            raise

    def compute_plan(self, target: Point) -> PolygonStamped:
        """
        Compute a needle path using geometric principles.

        Args:
            target (Point): Target point in 3D space

        Returns:
            PolygonStamped: Series of points representing the planned path

        Raises:
            ValueError: If target point is invalid
            RuntimeError: If planning fails
        """
        if not self.validate_target(target):
            raise ValueError("Invalid target point")

        try:
            # Compute circular arc path
            path_points = self._compute_circular_arc(target)

            # Apply smoothing if enabled
            if self._parameters['smoothing_enabled']:
                path_points = self._smooth_path(path_points)

            # Validate the generated path
            if not self.validate_path(path_points):
                raise RuntimeError("Generated path fails validation")

            return self._create_polygon_stamped(path_points)

        except Exception as e:
            self.log_error(f"Path planning failed: {str(e)}")
            raise RuntimeError(f"Path planning failed: {str(e)}")

    def _compute_circular_arc(self,
                            target: Point
                            ) -> List[Tuple[float, float, float]]:
        """
        Compute a circular arc path from origin to target.

        Args:
            target (Point): Target point

        Returns:
            List[Tuple[float, float, float]]: List of path points

        Raises:
            RuntimeError: If path computation fails
        """
        # Convert target to numpy array
        target_point = np.array([target.x, target.y, target.z])

        # Compute path parameters
        chord_length = np.linalg.norm(target_point)
        if chord_length < self._parameters['min_step']:
            raise RuntimeError("Target too close to insertion point")

        insertion_direction = np.array([0, 0, 1])  # Initial direction (Z-axis)

        # Compute rotation axis (cross product of insertion direction and target vector)
        rotation_axis = np.cross(insertion_direction, target_point)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm < 1e-6:
            # Target is along Z-axis, generate straight line
            return self._generate_straight_line(target_point)

        rotation_axis = rotation_axis / rotation_axis_norm

        # Compute rotation angle
        cos_theta = np.dot(insertion_direction, target_point) / chord_length
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # Check maximum angle constraint
        max_angle_rad = np.radians(self._parameters['max_angle'])
        if theta > max_angle_rad:
            raise RuntimeError(f"Required bend angle {np.degrees(theta):.1f}° "
                             f"exceeds maximum {self._parameters['max_angle']}°")

        # Generate arc points
        angles = np.linspace(0, theta, self._parameters['num_points'])
        path_points = []

        for angle in angles:
            # Use Rodrigues' rotation formula
            rotation_matrix = self._rodrigues_rotation(rotation_axis, angle)
            point = (chord_length * rotation_matrix @ insertion_direction)
            path_points.append((float(point[0]), float(point[1]), float(point[2])))

        return path_points

    def _generate_straight_line(self,
                              target_point: np.ndarray
                              ) -> List[Tuple[float, float, float]]:
        """Generate a straight line path for aligned targets."""
        t = np.linspace(0, 1, self._parameters['num_points'])
        points = []
        for ti in t:
            point = ti * target_point
            points.append((float(point[0]), float(point[1]), float(point[2])))
        return points

    def _rodrigues_rotation(self, axis: np.ndarray, theta: float) -> np.ndarray:
        """
        Compute rotation matrix using Rodrigues' rotation formula.

        Args:
            axis (np.ndarray): Rotation axis
            theta (float): Rotation angle in radians

        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = np.eye(3)
        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R

    def _smooth_path(self,
                    points: List[Tuple[float, float, float]]
                    ) -> List[Tuple[float, float, float]]:
        """
        Apply smoothing to the path points.

        Args:
            points (List[Tuple[float, float, float]]): Original path points

        Returns:
            List[Tuple[float, float, float]]: Smoothed path points
        """
        if len(points) < 3:
            return points

        points_array = np.array(points)
        window_size = min(self._parameters['smoothing_window'], len(points))

        if window_size % 2 == 0:
            window_size -= 1

        if window_size < 3:
            return points

        # Apply Savitzky-Golay filter for smoothing
        from scipy.signal import savgol_filter
        smoothed = np.zeros_like(points_array)

        # Keep endpoints unchanged
        smoothed[0] = points_array[0]
        smoothed[-1] = points_array[-1]

        # Smooth each dimension
        for i in range(3):
            smoothed[1:-1, i] = savgol_filter(
                points_array[:, i],
                window_size,
                3  # polynomial order
            )[1:-1]

        return [(p[0], p[1], p[2]) for p in smoothed]

    def validate_target(self, target: Point) -> bool:
        """
        Validate that the target point is reachable with allowable curvature.

        Args:
            target (Point): Target point to validate

        Returns:
            bool: True if target is valid, False otherwise
        """
        if not super().validate_target(target):
            return False

        # Check if required curvature is within limits
        target_point = np.array([target.x, target.y, target.z])
        chord_length = np.linalg.norm(target_point)

        if chord_length < 1e-6:  # Avoid division by zero
            self.log_error("Target coincides with insertion point")
            return False

        # Estimate required curvature from geometric properties
        deviation = np.sqrt(target.x**2 + target.y**2)  # Lateral deviation
        estimated_curvature = 2 * deviation / (chord_length**2)

        if estimated_curvature > self._parameters['max_curvature']:
            self.log_error(
                f"Required curvature {estimated_curvature:.4f} mm⁻¹ exceeds "
                f"maximum {self._parameters['max_curvature']:.4f} mm⁻¹")
            return False

        return True


def main():
    import rclpy
    from rclpy.node import Node

    class TestNode(Node):
        def __init__(self):
            super().__init__('test_geometric_planner')

    rclpy.init()
    node = TestNode()

    try:
        planner = GeometricPlanner(node)
        target = Point(x=0.0, y=0.0, z=2.0)

        try:
            plan = planner.compute_plan(target)
            node.get_logger().info(
                f"Generated plan with {len(plan.polygon.points)} points")

            # Test lateral deviation case
            target_lateral = Point(x=1.0, y=1.0, z=5.0)
            plan_lateral = planner.compute_plan(target_lateral)
            node.get_logger().info(
                f"Generated lateral deviation plan with "
                f"{len(plan_lateral.polygon.points)} points")

        except Exception as e:
            node.get_logger().error(f"Planning failed: {str(e)}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()