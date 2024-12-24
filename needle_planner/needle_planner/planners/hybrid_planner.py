#!/usr/bin/env python3

import os
from typing import List, Tuple, Optional

import numpy as np
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PolygonStamped
from sklearn.linear_model import Ridge
import pickle

from .base_planner import BasePlanner
from .geometric_planner import GeometricPlanner
from ..utils.fe_model import run_model


class HybridPlanner(BasePlanner):
    """
    Hybrid needle path planner.

    This planner combines geometric path planning with machine learning and
    finite element analysis for refinement. It first generates a geometric
    path, then uses ML to predict required corrections, and finally applies
    FE analysis for the final segment.
    """

    def __init__(self, node=None):
        """
        Initialize the hybrid planner.

        Args:
            node (Optional[Node]): ROS2 node for logging and parameters
        """
        super().__init__(node)

        # Load parameters
        self._load_parameters()

        # Initialize geometric planner
        self.geometric_planner = GeometricPlanner(node)

        # Initialize ML correction model
        self.correction_model = self._initialize_correction_model()

        # FE parameters (moved from default values to instance variables)
        self.fe_stiffness = np.array(self._parameters['stiffness_values'])
        self.fe_positions = np.array(self._parameters['position_values'])

    def _load_parameters(self) -> None:
        """Load planner-specific parameters."""
        if not self.node:
            # Set defaults if no node is provided
            self._parameters.update({
                'model_path': 'models/correction_model.pkl',
                'correction_scale': 0.1,
                'max_correction': 2.0,
                'segment_fraction': 0.25,
                'stiffness_values': [0.05, 0.06, 0.07],
                'position_values': [50.0, 75.0, 100.0],
                'guide_positions': [0.0, 0.0, 0.0]
            })
            return

        try:
            param_prefix = 'hybrid.'

            # ML correction parameters
            self.node.declare_parameter(
                f'{param_prefix}ml_correction.model_path',
                'models/correction_model.pkl')
            self.node.declare_parameter(
                f'{param_prefix}ml_correction.correction_scale', 0.1)
            self.node.declare_parameter(
                f'{param_prefix}ml_correction.max_correction', 2.0)

            # FE refinement parameters
            self.node.declare_parameter(
                f'{param_prefix}fe_refinement.segment_fraction', 0.25)
            self.node.declare_parameter(
                f'{param_prefix}fe_refinement.stiffness_values',
                [0.05, 0.06, 0.07])
            self.node.declare_parameter(
                f'{param_prefix}fe_refinement.position_values',
                [50.0, 75.0, 100.0])
            self.node.declare_parameter(
                f'{param_prefix}fe_refinement.guide_positions',
                [0.0, 0.0, 0.0])

            # Update parameters
            self._parameters.update({
                'model_path': self.node.get_parameter(
                    f'{param_prefix}ml_correction.model_path').value,
                'correction_scale': self.node.get_parameter(
                    f'{param_prefix}ml_correction.correction_scale').value,
                'max_correction': self.node.get_parameter(
                    f'{param_prefix}ml_correction.max_correction').value,
                'segment_fraction': self.node.get_parameter(
                    f'{param_prefix}fe_refinement.segment_fraction').value,
                'stiffness_values': self.node.get_parameter(
                    f'{param_prefix}fe_refinement.stiffness_values').value,
                'position_values': self.node.get_parameter(
                    f'{param_prefix}fe_refinement.position_values').value,
                'guide_positions': self.node.get_parameter(
                    f'{param_prefix}fe_refinement.guide_positions').value,
            })

        except Exception as e:
            self.log_error(f"Failed to load parameters: {str(e)}")
            raise

    def _initialize_correction_model(self) -> Ridge:
        """
        Initialize the ML correction model.

        Returns:
            Ridge: Initialized correction model
        """
        try:
            pkg_share = get_package_share_directory('needle_planner')
            model_path = os.path.join(
                pkg_share, 'models',
                os.path.basename(self._parameters['model_path'])
            )

            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.log_info(f"Loaded correction model from {model_path}")
                return model
            else:
                self.log_warn(f"No pre-trained model found at {model_path}. "
                            "Using dummy model.")
                # Initialize a dummy model for demonstration
                model = Ridge(alpha=1.0)
                # Note: In practice, you should load a properly trained model
                return model

        except Exception as e:
            self.log_error(f"Failed to load correction model: {str(e)}")
            raise

    def compute_plan(self, target: Point) -> PolygonStamped:
        """
        Compute a needle path using hybrid planning approach.

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
            # 1. Generate initial geometric path
            initial_plan = self.geometric_planner.compute_plan(target)
            initial_points = [(p.x, p.y, p.z) for p in initial_plan.polygon.points]

            # 2. Apply ML-based corrections
            corrected_points = self._apply_ml_corrections(initial_points)

            # 3. Use FE analysis for final segment refinement
            final_points = self._refine_final_segment(corrected_points, target)

            # Validate the final path
            if not self.validate_path(final_points):
                raise RuntimeError("Final path fails validation")

            return self._create_polygon_stamped(final_points)

        except Exception as e:
            self.log_error(f"Path planning failed: {str(e)}")
            raise RuntimeError(f"Path planning failed: {str(e)}")

    def _apply_ml_corrections(self,
                            points: List[Tuple[float, float, float]]
                            ) -> List[Tuple[float, float, float]]:
        """
        Apply ML-predicted corrections to the path points.

        Args:
            points: List of (x,y,z) points

        Returns:
            List of corrected (x,y,z) points
        """
        points_array = np.array(points)

        try:
            # Generate features for correction prediction
            features = self._generate_correction_features(points_array)

            # Predict corrections using the ML model
            corrections = self.correction_model.predict(features)

            # Scale corrections and apply limits
            corrections *= self._parameters['correction_scale']
            corrections = np.clip(corrections,
                                -self._parameters['max_correction'],
                                self._parameters['max_correction'])

            # Apply corrections while preserving path continuity
            corrected_points = points_array + corrections.reshape(-1, 3)

            # Ensure the path starts at the origin
            corrected_points[0] = [0, 0, 0]

            # Ensure smooth transitions
            return self._smooth_corrections(
                points_array,
                corrected_points.reshape(-1, 3)
            )

        except Exception as e:
            self.log_error(f"ML correction failed: {str(e)}")
            return points

    def _generate_correction_features(self, points: np.ndarray) -> np.ndarray:
        """
        Generate features for ML correction.

        Args:
            points (np.ndarray): Path points array

        Returns:
            np.ndarray: Features for ML model
        """
        # Calculate path properties for features
        tangents = np.diff(points, axis=0)
        tangents = np.vstack([tangents, tangents[-1]])  # Repeat last tangent

        # Normalize tangents
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1
        tangents = tangents / norms

        # Calculate curvature approximation
        curvature = np.zeros(len(points))
        for i in range(1, len(points)-1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            angle = np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                -1.0, 1.0
            ))
            curvature[i] = angle / np.linalg.norm(v1)

        # Combine features
        features = np.hstack([
            points,  # Position
            tangents,  # Direction
            curvature.reshape(-1, 1),  # Local curvature
            np.linalg.norm(points, axis=1).reshape(-1, 1)  # Distance from origin
        ])

        return features

    def _smooth_corrections(self,
                          original: np.ndarray,
                          corrected: np.ndarray,
                          window_size: int = 5
                          ) -> List[Tuple[float, float, float]]:
        """
        Smooth the transition between original and corrected points.

        Args:
            original (np.ndarray): Original path points
            corrected (np.ndarray): Corrected path points
            window_size (int): Smoothing window size

        Returns:
            List[Tuple[float, float, float]]: Smoothed path points
        """
        if window_size % 2 == 0:
            window_size += 1

        from scipy.signal import savgol_filter

        # Calculate corrections
        corrections = corrected - original

        # Smooth corrections
        smoothed_corrections = np.zeros_like(corrections)
        for i in range(3):  # For each dimension
            smoothed_corrections[:, i] = savgol_filter(
                corrections[:, i],
                window_size,
                3  # polynomial order
            )

        # Apply smoothed corrections
        smoothed_points = original + smoothed_corrections

        # Ensure start point remains at origin
        smoothed_points[0] = [0, 0, 0]

        return [(p[0], p[1], p[2]) for p in smoothed_points]

    def _refine_final_segment(self,
                            points: List[Tuple[float, float, float]],
                            target: Point
                            ) -> List[Tuple[float, float, float]]:
        """
        Refine the final segment using FE analysis.

        Args:
            points: List of path points
            target: Target point

        Returns:
            List of refined path points
        """
        # Extract the final segment (last 25% of points)
        segment_start = int(len(points) * (1 - self._parameters['segment_fraction']))
        final_segment = points[segment_start:]

        try:
            # Prepare FE inputs
            tip_pos = np.array([target.x, target.y, target.z])
            num_points = len(final_segment)

            # Initialize arrays for FE analysis
            y_rxn = np.zeros(num_points)  # Reaction forces
            y_disp = np.array([p[1] for p in final_segment])  # Current Y displacements
            guide_pos_vec = np.array(self._parameters['guide_positions'])
            guide_pos_ins = np.zeros(num_points)  # Guide insertion positions

            # Run FE analysis
            y_disp_new, y_rxn_new, tip_path = run_model(
                tip_pos=tip_pos,
                y_rxn=y_rxn,
                y_disp=y_disp,
                stiffness_vec=self.fe_stiffness,
                pos_vec=self.fe_positions,
                tip_path=[],
                guide_pos_vec=guide_pos_vec,
                guide_pos_ins=guide_pos_ins
            )

            # Generate refined path
            refined_final_segment = []
            x_positions = np.linspace(
                final_segment[0][0],
                target.x,
                num_points
            )
            z_positions = np.linspace(
                final_segment[0][2],
                target.z,
                num_points
            )

            for i in range(num_points):
                refined_final_segment.append(
                    (x_positions[i], float(y_disp_new[i]), z_positions[i])
                )

            # Combine original points with refined segment
            return points[:segment_start] + refined_final_segment

        except Exception as e:
            self.log_error(f"FE refinement failed: {str(e)}. Using original path.")
            return points


def main():
    import rclpy
    from rclpy.node import Node

    class TestNode(Node):
        def __init__(self):
            super().__init__('test_hybrid_planner')

    rclpy.init()
    node = TestNode()

    try:
        planner = HybridPlanner(node)
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