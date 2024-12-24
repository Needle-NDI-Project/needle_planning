#!/usr/bin/env python3

import os
import pickle
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PolygonStamped
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .base_planner import BasePlanner

@dataclass
class PretrainedModel:
    """Container for pretrained model components."""
    regressor: any
    pca: PCA
    poly_features: PolynomialFeatures
    scaler: StandardScaler
    config: Dict

class LinearRegressionPlanner(BasePlanner):
    """
    Linear regression-based needle path planner.

    This planner uses pre-trained PCA and polynomial regression for path generation.
    All transformers (PCA, PolynomialFeatures, StandardScaler) are fitted during
    training and only used for transformation during inference.
    """

    def __init__(self, node=None):
        """
        Initialize the linear regression planner.

        Args:
            node (Optional[Node]): ROS2 node for logging and parameters
        """
        super().__init__(node)

        # Load parameters
        self._load_parameters()

        # Load the pre-trained model components
        self.model = self._load_model()

        if self.model is None:
            raise RuntimeError("Failed to load regression model and components")

    def _load_parameters(self) -> None:
        """Load planner-specific parameters."""
        if not self.node:
            # Set defaults if no node is provided
            self._parameters.update({
                'model_path': 'models/linear_regression_model.pkl',
                'num_points': 50,
                'spacing': 0.5,  # mm
                'interpolation_method': 'cubic'
            })
            return

        try:
            # Declare and get parameters
            param_prefix = 'linear_regression.'
            self.node.declare_parameter(
                f'{param_prefix}model.path',
                'models/linear_regression_model.pkl')
            self.node.declare_parameter(
                f'{param_prefix}path_generation.num_points', 50)
            self.node.declare_parameter(
                f'{param_prefix}path_generation.spacing', 0.5)
            self.node.declare_parameter(
                f'{param_prefix}path_generation.interpolation', 'cubic')

            # Update parameters
            self._parameters.update({
                'model_path': self.node.get_parameter(
                    f'{param_prefix}model.path').value,
                'num_points': self.node.get_parameter(
                    f'{param_prefix}path_generation.num_points').value,
                'spacing': self.node.get_parameter(
                    f'{param_prefix}path_generation.spacing').value,
                'interpolation_method': self.node.get_parameter(
                    f'{param_prefix}path_generation.interpolation').value
            })

        except Exception as e:
            self.log_error(f"Failed to load parameters: {str(e)}")
            raise

    def _load_model(self) -> Optional[PretrainedModel]:
        """
        Load the pre-trained model and its components.

        Returns:
            Optional[PretrainedModel]: Container with model and transformers
        """
        try:
            pkg_share = get_package_share_directory('needle_planner')
            model_path = os.path.join(
                pkg_share, 'models',
                os.path.basename(self._parameters['model_path']))

            if not os.path.exists(model_path):
                self.log_error(f"Model file not found: {model_path}")
                return None

            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)

            # Verify all components are present
            required_keys = ['regressor', 'pca', 'poly_features', 'scaler', 'config']
            if not all(key in model_data for key in required_keys):
                self.log_error("Model file missing required components")
                return None

            return PretrainedModel(
                regressor=model_data['regressor'],
                pca=model_data['pca'],
                poly_features=model_data['poly_features'],
                scaler=model_data['scaler'],
                config=model_data['config']
            )

        except Exception as e:
            self.log_error(f"Failed to load model: {str(e)}")
            return None

    def compute_plan(self, target: Point) -> PolygonStamped:
        """
        Compute a needle path plan using the pre-trained model.

        Args:
            target (Point): Target point in 3D space

        Returns:
            PolygonStamped: Series of points representing the planned path

        Raises:
            ValueError: If target is invalid
            RuntimeError: If planning fails
        """
        if not self.validate_target(target):
            raise ValueError("Invalid target point")

        try:
            # Generate intermediate points for path interpolation
            points = self._generate_interpolation_points(target)

            # Transform points using pre-trained PCA (no fitting)
            points_3d = np.array(points)
            try:
                points_2d = self.model.pca.transform(points_3d)
            except ValueError as e:
                self.log_error(f"PCA transform failed: {str(e)}")
                raise RuntimeError("PCA transformation failed")

            # Generate path points
            path_points = self._generate_path_points(target.z)

            # Validate the generated path
            if not self.validate_path(path_points):
                raise RuntimeError("Generated path fails validation")

            return self._create_polygon_stamped(path_points)

        except Exception as e:
            self.log_error(f"Path planning failed: {str(e)}")
            raise RuntimeError(f"Path planning failed: {str(e)}")

    def _generate_path_points(self, end_z: float) -> List[Tuple[float, float, float]]:
        """
        Generate a series of points along the planned path.

        Args:
            end_z (float): Z-coordinate of the target point

        Returns:
            List[Tuple[float, float, float]]: List of path points
        """
        # Generate points in z-direction with proper spacing
        z_points = np.arange(0, end_z + self._parameters['spacing'],
                           self._parameters['spacing'])
        out_z = z_points.reshape(-1, 1)

        # Initialize path points
        out_3d = np.column_stack((
            np.zeros(out_z.shape),
            np.zeros(out_z.shape),
            out_z
        ))

        try:
            # Transform using pre-trained components (no fitting)
            out_pca = self.model.pca.transform(out_3d)
            x_pca = out_pca[:, 0]

            # Generate polynomial features
            x_poly = self.model.poly_features.transform(x_pca.reshape(-1, 1))

            # Scale features if scaler exists
            if self.model.scaler is not None:
                x_poly = self.model.scaler.transform(x_poly)

            # Use the model for prediction
            y_pred = self.model.regressor.predict(x_poly)

            # Transform back to 3D
            path_pca = np.column_stack((x_pca, y_pred))
            path_3d = self.model.pca.inverse_transform(path_pca)

            # Ensure proper spacing and smoothness
            return self._post_process_path(path_3d)

        except Exception as e:
            self.log_error(f"Path generation failed: {str(e)}")
            raise RuntimeError("Failed to generate path points")

    def _post_process_path(self,
                          path_points: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Post-process the generated path points.

        Args:
            path_points (np.ndarray): Raw path points

        Returns:
            List[Tuple[float, float, float]]: Processed path points
        """
        # Ensure start point is at origin
        path_points[0] = [0, 0, 0]

        # Apply smoothing if needed
        if len(path_points) > 3:
            from scipy.signal import savgol_filter

            window_size = min(7, len(path_points) // 2)
            if window_size % 2 == 0:
                window_size -= 1

            if window_size >= 3:
                try:
                    for i in range(3):  # Smooth x, y, z separately
                        path_points[:, i] = savgol_filter(
                            path_points[:, i],
                            window_size,
                            3  # polynomial order
                        )
                except Exception as e:
                    self.log_warn(f"Smoothing failed, using unsmoothed path: {str(e)}")

        # Ensure proper spacing
        from ..utils.path_utils import interpolate_path
        points_list = [(p[0], p[1], p[2]) for p in path_points]
        return interpolate_path(
            points_list,
            self._parameters['num_points'],
            method=self._parameters['interpolation_method']
        )

def main():
    import rclpy
    from rclpy.node import Node

    class TestNode(Node):
        def __init__(self):
            super().__init__('test_linear_regression_planner')

    rclpy.init()
    node = TestNode()

    try:
        planner = LinearRegressionPlanner(node)
        target = Point(x=0.0, y=0.0, z=2.0)

        try:
            plan = planner.compute_plan(target)
            node.get_logger().info(
                f"Generated plan with {len(plan.polygon.points)} points")
        except Exception as e:
            node.get_logger().error(f"Planning failed: {str(e)}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()