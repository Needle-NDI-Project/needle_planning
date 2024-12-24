#!/usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Point, PolygonStamped
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from scipy.interpolate import splprep, splev

from .base_planner import BasePlanner

@dataclass
class Node:
    """Tree node for RRT-like planning."""
    position: np.ndarray
    parent: Optional['Node'] = None
    children: List['Node'] = None
    cost: float = 0.0
    curvature: float = 0.0  # Store local curvature

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, child: 'Node') -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent = self

class SamplingPlanner(BasePlanner):
    """
    Sampling-based needle path planner.

    This planner uses a modified RRT approach adapted for needle steering,
    with improved curvature calculation and path validation.
    """

    def __init__(self, node=None):
        """
        Initialize the sampling-based planner.

        Args:
            node (Optional[Node]): ROS2 node for logging and parameters
        """
        super().__init__(node)
        self._load_parameters()
        self.workspace_bounds = self._compute_workspace_bounds()

    def _load_parameters(self) -> None:
        """Load planner-specific parameters."""
        if not self.node:
            self._parameters.update({
                'max_iterations': 1000,
                'step_size': 2.0,  # mm
                'goal_bias': 0.2,
                'max_curvature': 0.01,  # 1/mm
                'goal_threshold': 2.0,  # mm
                'min_step': 0.5,  # mm
                'spline_smoothing': 0.1,
                'optimization_time': 0.5,  # seconds
                'max_extension_attempts': 10,
                'curvature_weight': 0.7,  # Weight for curvature vs distance
                'max_angle': np.pi/4  # Maximum allowed angle between segments
            })
            return

        try:
            param_prefix = 'sampling.planning.'

            # Planning parameters
            self.node.declare_parameter(f'{param_prefix}max_iterations', 1000)
            self.node.declare_parameter(f'{param_prefix}step_size', 2.0)
            self.node.declare_parameter(f'{param_prefix}goal_bias', 0.2)
            self.node.declare_parameter(f'{param_prefix}max_curvature', 0.01)
            self.node.declare_parameter(f'{param_prefix}goal_threshold', 2.0)
            self.node.declare_parameter(f'{param_prefix}min_step', 0.5)
            self.node.declare_parameter(f'{param_prefix}spline_smoothing', 0.1)
            self.node.declare_parameter(f'{param_prefix}optimization_time', 0.5)
            self.node.declare_parameter(f'{param_prefix}max_extension_attempts', 10)
            self.node.declare_parameter(f'{param_prefix}curvature_weight', 0.7)
            self.node.declare_parameter(f'{param_prefix}max_angle', np.pi/4)

            # Update parameters
            for param in self._parameters:
                self._parameters[param] = self.node.get_parameter(
                    f'{param_prefix}{param}').value

        except Exception as e:
            self.log_error(f"Failed to load parameters: {str(e)}")
            raise

    def compute_circular_curvature(self,
                                 p1: np.ndarray,
                                 p2: np.ndarray,
                                 p3: np.ndarray) -> float:
        """
        Compute curvature using circumscribed circle method.

        Args:
            p1, p2, p3: Three consecutive points

        Returns:
            float: Computed curvature (1/radius)
        """
        # Compute vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Compute vector lengths
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)

        if l1 < 1e-6 or l2 < 1e-6:
            return float('inf')

        # Compute angle between vectors
        cos_angle = np.dot(v1, v2) / (l1 * l2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Compute radius and curvature
        if angle < 1e-6:
            return 0.0

        # Use circumscribed circle formula
        a = l1
        b = l2
        c = np.linalg.norm(p3 - p1)

        if c < 1e-6:
            return float('inf')

        s = (a + b + c) / 2  # Semi-perimeter
        try:
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            radius = (a * b * c) / (4 * area)
            return 1.0 / radius
        except:
            return float('inf')

    def compute_spline_curvature(self, points: List[np.ndarray]) -> np.ndarray:
        """
        Compute curvature using B-spline approximation.

        Args:
            points: List of path points

        Returns:
            np.ndarray: Array of curvature values
        """
        if len(points) < 3:
            return np.zeros(len(points))

        try:
            # Convert points to array
            points_array = np.array(points)

            # Fit B-spline
            tck, u = splprep([points_array[:, i] for i in range(3)],
                           s=self._parameters['spline_smoothing'])

            # Compute derivatives
            deriv1 = np.array(splev(u, tck, der=1))
            deriv2 = np.array(splev(u, tck, der=2))

            # Calculate curvature using Frenet formula
            cross_prod = np.cross(deriv1.T, deriv2.T)
            norms = np.linalg.norm(cross_prod, axis=1)
            velocity_norms = np.linalg.norm(deriv1, axis=0)

            # Avoid division by zero
            mask = velocity_norms > 1e-6
            curvature = np.zeros(len(points))
            curvature[mask] = norms[mask] / (velocity_norms[mask] ** 3)

            return curvature

        except Exception as e:
            self.log_warn(f"Spline curvature calculation failed: {str(e)}")
            # Fallback to simpler method
            return self._compute_discrete_curvature(points)

    def _compute_discrete_curvature(self, points: List[np.ndarray]) -> np.ndarray:
        """
        Compute discrete curvature using three consecutive points.

        Args:
            points: List of path points

        Returns:
            np.ndarray: Array of curvature values
        """
        curvature = np.zeros(len(points))

        for i in range(1, len(points) - 1):
            curvature[i] = self.compute_circular_curvature(
                points[i-1], points[i], points[i+1])

        # Set end points curvature same as neighbors
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

        return curvature

    def _check_extension_valid(self,
                             pos1: np.ndarray,
                             pos2: np.ndarray,
                             use_circular_approx: bool = True) -> bool:
        """
        Check if the path segment satisfies constraints.

        Args:
            pos1: Start position
            pos2: End position
            use_circular_approx: Whether to use circular approximation

        Returns:
            bool: True if extension is valid
        """
        # Check workspace bounds
        for i, (min_val, max_val) in enumerate([
            self.workspace_bounds['x'],
            self.workspace_bounds['y'],
            self.workspace_bounds['z']
        ]):
            if not (min_val <= pos2[i] <= max_val):
                return False

        # Handle start point case
        if np.linalg.norm(pos1) < 1e-6:
            return True

        # Compute segment properties
        direction = pos2 - pos1
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return False

        # Check angle constraint
        if hasattr(self, '_prev_direction'):
            angle = np.arccos(np.clip(
                np.dot(direction, self._prev_direction) /
                (length * np.linalg.norm(self._prev_direction)),
                -1.0, 1.0
            ))
            if abs(angle) > self._parameters['max_angle']:
                return False

        self._prev_direction = direction

        # Check curvature constraint
        if use_circular_approx:
            # Use circular approximation for quick check
            deviation = np.sqrt(pos2[0]**2 + pos2[1]**2)
            estimated_curvature = 2 * deviation / (length**2)
            return estimated_curvature <= self._parameters['max_curvature']
        else:
            # Use more accurate curvature calculation for final validation
            # Get recent points in path
            if hasattr(self, '_path_points'):
                points = self._path_points[-3:] + [pos2]
                if len(points) >= 3:
                    curvature = self.compute_spline_curvature(points)
                    return np.all(curvature <= self._parameters['max_curvature'])
            return True

    def _extend_towards(self,
                       from_pos: np.ndarray,
                       to_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Extend tree from current position towards target position.

        Args:
            from_pos: Current position
            to_pos: Target position

        Returns:
            Optional[np.ndarray]: New position if extension is valid
        """
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return None

        # Try multiple step sizes if initial extension fails
        for attempt in range(self._parameters['max_extension_attempts']):
            # Reduce step size with each attempt
            reduction_factor = 1.0 / (attempt + 1)
            step = min(self._parameters['step_size'] * reduction_factor, distance)

            # Calculate new position
            new_direction = direction / distance
            new_pos = from_pos + new_direction * step

            # Check if extension is valid
            if self._check_extension_valid(from_pos, new_pos):
                return new_pos

        return None

    def _optimize_path(self,
                      path_points: List[np.ndarray],
                      target: np.ndarray) -> List[np.ndarray]:
        """
        Apply optimization to the path.

        Args:
            path_points: List of path points
            target: Target point

        Returns:
            List[np.ndarray]: Optimized path points
        """
        if len(path_points) < 3:
            return path_points

        # Create a copy for optimization
        optimized = path_points.copy()

        # Calculate initial curvature
        initial_curvature = self.compute_spline_curvature(optimized)

        # Iterative optimization
        for i in range(1, len(optimized) - 1):
            best_pos = optimized[i]
            min_cost = float('inf')

            # Sample positions around current point
            for _ in range(10):
                # Generate random perturbation
                perturbation = np.random.randn(3) * self._parameters['step_size'] * 0.1
                test_pos = optimized[i] + perturbation

                # Check if perturbation is valid
                if not self._check_extension_valid(optimized[i-1], test_pos) or \
                   not self._check_extension_valid(test_pos, optimized[i+1]):
                    continue

                # Calculate cost
                test_points = optimized.copy()
                test_points[i] = test_pos
                curvature = self.compute_spline_curvature(test_points)

                # Weighted cost: curvature + distance to target
                curvature_cost = np.max(curvature)
                target_dist = np.linalg.norm(test_pos - target)

                total_cost = (self._parameters['curvature_weight'] * curvature_cost +
                            (1 - self._parameters['curvature_weight']) * target_dist)

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_pos = test_pos

            optimized[i] = best_pos

        return optimized

    def compute_plan(self, target: Point) -> PolygonStamped:
        """
        Compute a needle path using sampling-based planning.

        Args:
            target: Target point in 3D space

        Returns:
            PolygonStamped: Series of points representing the planned path

        Raises:
            ValueError: If target point is invalid
            RuntimeError: If planning fails
        """
        if not self.validate_target(target):
            raise ValueError("Invalid target point")

        try:
            # Convert target to numpy array
            target_point = np.array([target.x, target.y, target.z])

            # Initialize storage for path points
            self._path_points = []
            self._prev_direction = None

            # Initialize tree with start node
            start_node = Node(np.array([0.0, 0.0, 0.0]))

            # Grow the tree
            final_node = self._grow_tree(start_node, target_point)

            if final_node is None:
                raise RuntimeError("Failed to find a valid path")

            # Extract and optimize the path
            path_points = self._extract_and_optimize_path(final_node, target_point)

            # Validate the final path
            if not self.validate_path(path_points):
                raise RuntimeError("Generated path fails validation")

            # Clear temporary storage
            self._path_points = []
            self._prev_direction = None

            return self._create_polygon_stamped(path_points)

        except Exception as e:
            self.log_error(f"Path planning failed: {str(e)}")
            raise RuntimeError(f"Path planning failed: {str(e)}")

    def _grow_tree(self,
                   start_node: Node,
                   target: np.ndarray,
                   max_retries: int = 3) -> Optional[Node]:
        """
        Grow the RRT tree towards the target.

        Args:
            start_node: Root node of the tree
            target: Target point
            max_retries: Maximum number of planning attempts

        Returns:
            Optional[Node]: Final node if path is found
        """
        for retry in range(max_retries):
            nodes = [start_node]
            self._path_points = []  # Reset path points

            for i in range(self._parameters['max_iterations']):
                # Sample random point (with goal bias)
                if np.random.random() < self._parameters['goal_bias']:
                    sample = target
                else:
                    sample = self._sample_random_point()

                # Find nearest node considering both distance and curvature
                nearest_node = self._find_best_parent(nodes, sample)

                # Extend tree towards sample
                new_position = self._extend_towards(nearest_node.position, sample)

                if new_position is not None:
                    # Create new node and update path points
                    new_node = Node(new_position)
                    nearest_node.add_child(new_node)
                    nodes.append(new_node)
                    self._path_points.append(new_position)

                    # Check if we reached the target
                    if np.linalg.norm(new_position - target) < self._parameters['goal_threshold']:
                        self.log_info(f"Found path after {i+1} iterations (attempt {retry+1})")
                        return new_node

            self.log_warn(f"Planning attempt {retry+1} failed")

        self.log_error("Max retry attempts reached without finding path")
        return None

    def _find_best_parent(self,
                         nodes: List[Node],
                         sample: np.ndarray) -> Node:
        """
        Find the best parent node considering distance and curvature.

        Args:
            nodes: List of available nodes
            sample: Target sample point

        Returns:
            Node: Best parent node
        """
        best_node = nodes[0]
        best_cost = float('inf')

        for node in nodes:
            # Calculate Euclidean distance
            distance = np.linalg.norm(node.position - sample)

            # Calculate curvature cost if we have enough points
            curvature_cost = 0.0
            if len(self._path_points) >= 2:
                points = self._path_points[-2:] + [node.position, sample]
                curvature = self.compute_spline_curvature(points)
                curvature_cost = np.max(curvature)

            # Weighted cost
            total_cost = (1 - self._parameters['curvature_weight']) * distance + \
                        self._parameters['curvature_weight'] * curvature_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_node = node

        return best_node

    def _sample_random_point(self) -> np.ndarray:
        """
        Generate a random point in the workspace.

        Returns:
            np.ndarray: Random point coordinates
        """
        while True:
            point = np.array([
                np.random.uniform(self.workspace_bounds['x'][0],
                                self.workspace_bounds['x'][1]),
                np.random.uniform(self.workspace_bounds['y'][0],
                                self.workspace_bounds['y'][1]),
                np.random.uniform(self.workspace_bounds['z'][0],
                                self.workspace_bounds['z'][1])
            ])

            # Basic validation
            if np.linalg.norm(point[:2]) <= self._parameters['max_lateral']:
                return point

    def _extract_and_optimize_path(self,
                                 final_node: Node,
                                 target: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Extract and optimize the path from the tree.

        Args:
            final_node: Final node of the path
            target: Target point

        Returns:
            List[Tuple[float, float, float]]: Optimized path points
        """
        # Extract initial path
        path_points = []
        current_node = final_node

        while current_node is not None:
            path_points.append(current_node.position)
            current_node = current_node.parent

        path_points = list(reversed(path_points))

        # Optimize path
        start_time = time.time()
        while time.time() - start_time < self._parameters['optimization_time']:
            path_points = self._optimize_path(path_points, target)

        # Smooth the path using splines
        try:
            # Fit spline to points
            points_array = np.array(path_points)
            tck, u = splprep([points_array[:, i] for i in range(3)],
                           s=self._parameters['spline_smoothing'])

            # Generate more points along spline
            u_new = np.linspace(0, 1, self._parameters['num_points'])
            smoothed = np.array(splev(u_new, tck)).T

            # Ensure start and end points are preserved
            smoothed[0] = path_points[0]
            smoothed[-1] = path_points[-1]

            path_points = smoothed

        except Exception as e:
            self.log_warn(f"Spline smoothing failed, using original path: {str(e)}")

        # Convert to list of tuples
        return [(float(p[0]), float(p[1]), float(p[2])) for p in path_points]


def main():
    import rclpy
    from rclpy.node import Node

    class TestNode(Node):
        def __init__(self):
            super().__init__('test_sampling_planner')

    rclpy.init()
    node = TestNode()

    try:
        planner = SamplingPlanner(node)
        target = Point(x=0.0, y=0.0, z=2.0)

        try:
            plan = planner.compute_plan(target)
            node.get_logger().info(
                f"Generated plan with {len(plan.polygon.points)} points")

            # Test lateral deviation case
            target_lateral = Point(x=1.0, y=1.0, z=5.0)
            plan_lateral = planner.compute_plan(target_lateral)
            node.get_logger().info(
                f"Generated lateral plan with {len(plan_lateral.polygon.points)} points")

        except Exception as e:
            node.get_logger().error(f"Planning failed: {str(e)}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()