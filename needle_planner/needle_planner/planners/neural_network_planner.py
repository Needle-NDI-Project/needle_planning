#!/usr/bin/env python3

import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PolygonStamped


from .base_planner import BasePlanner

class NeedlePathNN(nn.Module):
    """
    Neural network architecture for needle path planning.

    This network takes a target point as input and produces a sequence
    of waypoints forming a feasible needle path. It includes safety
    checks and numerical stability improvements.
    """
    def __init__(self,
                 input_dim: int = 3,
                 hidden_layers: List[int] = None,
                 output_points: int = 20,
                 dropout_rate: float = 0.1,
                 min_scale: float = 1e-6):
        """
        Initialize the network architecture.

        Args:
            input_dim: Dimension of input (target point)
            hidden_layers: List of hidden layer sizes
            output_points: Number of waypoints to generate
            dropout_rate: Dropout probability
            min_scale: Minimum scaling factor for numerical stability
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [32, 64, 128]

        self.min_scale = min_scale
        self.output_points = output_points

        # Input embedding
        layers = []
        prev_size = input_dim

        for size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(size)  # Add BatchNorm for stability
            ])
            prev_size = size

        self.encoder = nn.Sequential(*layers)

        # Path generation decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_layers[-1] * 2),
            nn.Linear(hidden_layers[-1] * 2, output_points * 3),
            nn.Tanh()  # Bound outputs to [-1, 1]
        )

        # Learnable scale factor
        self.scale_factor = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_points, 3)
        """
        # Input validation
        if torch.any(torch.isnan(x)):
            raise ValueError("Input contains NaN values")

        # Encode input
        features = self.encoder(x)

        # Generate path points
        path = self.decoder(features)
        path = path.view(-1, self.output_points, 3)

        # Safe path scaling
        target_dist = torch.norm(x, dim=1, keepdim=True)
        path_norms = torch.norm(path, dim=2)
        max_norm = torch.max(path_norms, dim=1, keepdim=True)[0]

        # Add small epsilon to prevent division by zero
        max_norm = torch.clamp(max_norm, min=self.min_scale)

        # Scale path to match target distance
        scale = target_dist.unsqueeze(1) / max_norm
        path = path * scale.unsqueeze(2)

        # Ensure first point is at origin
        path = torch.cat([
            torch.zeros_like(path[:, :1, :]),
            path[:, 1:, :]
        ], dim=1)

        return path


class NeuralNetworkPlanner(BasePlanner):
    """
    Neural network-based needle path planner.

    This planner uses a deep neural network to generate smooth needle paths,
    with improved numerical stability and error handling.
    """

    def __init__(self, node=None):
        """
        Initialize the neural network planner.

        Args:
            node (Optional[Node]): ROS2 node for logging and parameters
        """
        super().__init__(node)

        # Load parameters
        self._load_parameters()

        # Initialize the neural network
        self.model = self._create_model()
        self.model.eval()  # Set to evaluation mode

        # Load pre-trained weights if available
        self._load_weights()

        # Set device and move model
        self.device = torch.device(self._parameters['device'])
        self.model = self.model.to(self.device)

    def _load_parameters(self) -> None:
        """Load planner-specific parameters."""
        if not self.node:
            # Set defaults if no node is provided
            self._parameters.update({
                'input_dim': 3,
                'hidden_layers': [32, 64, 128],
                'output_points': 20,
                'dropout_rate': 0.1,
                'min_scale': 1e-6,
                'device': 'cpu',
                'batch_size': 1,
                'checkpoint_path': 'models/needle_nn_model.pt',
                'confidence_threshold': 0.8
            })
            return

        try:
            param_prefix = 'neural_network.'

            # Model architecture parameters
            self.node.declare_parameter(
                f'{param_prefix}model.architecture.input_dim', 3)
            self.node.declare_parameter(
                f'{param_prefix}model.architecture.hidden_layers', [32, 64, 128])
            self.node.declare_parameter(
                f'{param_prefix}model.architecture.output_points', 20)
            self.node.declare_parameter(
                f'{param_prefix}model.architecture.dropout_rate', 0.1)
            self.node.declare_parameter(
                f'{param_prefix}model.architecture.min_scale', 1e-6)

            # Inference parameters
            self.node.declare_parameter(
                f'{param_prefix}inference.batch_size', 1)
            self.node.declare_parameter(
                f'{param_prefix}inference.device', 'cpu')
            self.node.declare_parameter(
                f'{param_prefix}inference.checkpoint_path',
                'models/needle_nn_model.pt')
            self.node.declare_parameter(
                f'{param_prefix}inference.confidence_threshold', 0.8)

            # Update parameters
            self._parameters.update({
                'input_dim': self.node.get_parameter(
                    f'{param_prefix}model.architecture.input_dim').value,
                'hidden_layers': self.node.get_parameter(
                    f'{param_prefix}model.architecture.hidden_layers').value,
                'output_points': self.node.get_parameter(
                    f'{param_prefix}model.architecture.output_points').value,
                'dropout_rate': self.node.get_parameter(
                    f'{param_prefix}model.architecture.dropout_rate').value,
                'min_scale': self.node.get_parameter(
                    f'{param_prefix}model.architecture.min_scale').value,
                'batch_size': self.node.get_parameter(
                    f'{param_prefix}inference.batch_size').value,
                'device': self.node.get_parameter(
                    f'{param_prefix}inference.device').value,
                'checkpoint_path': self.node.get_parameter(
                    f'{param_prefix}inference.checkpoint_path').value,
                'confidence_threshold': self.node.get_parameter(
                    f'{param_prefix}inference.confidence_threshold').value
            })

        except Exception as e:
            self.log_error(f"Failed to load parameters: {str(e)}")
            raise

    def _create_model(self) -> NeedlePathNN:
        """
        Create the neural network model.

        Returns:
            NeedlePathNN: Initialized model
        """
        return NeedlePathNN(
            input_dim=self._parameters['input_dim'],
            hidden_layers=self._parameters['hidden_layers'],
            output_points=self._parameters['output_points'],
            dropout_rate=self._parameters['dropout_rate'],
            min_scale=self._parameters['min_scale']
        )

    def _load_weights(self) -> None:
        """Load pre-trained model weights if available."""
        try:
            pkg_share = get_package_share_directory('needle_planner')
            checkpoint_path = os.path.join(
                pkg_share, 'models',
                os.path.basename(self._parameters['checkpoint_path'])
            )

            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                # Verify state dict compatibility
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    state_dict, strict=False)

                if missing_keys:
                    self.log_warn(f"Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    self.log_warn(f"Unexpected keys in checkpoint: {unexpected_keys}")

                self.log_info(f"Loaded model weights from {checkpoint_path}")
            else:
                self.log_warn(f"No pre-trained weights found at {checkpoint_path}")

        except Exception as e:
            self.log_error(f"Failed to load model weights: {str(e)}")
            raise

    def compute_plan(self, target: Point) -> PolygonStamped:
        """
        Compute a needle path using the neural network.

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
            # Convert target to tensor with input validation
            target_point = np.array([target.x, target.y, target.z])
            if np.any(np.isnan(target_point)):
                raise ValueError("Target point contains NaN values")

            target_tensor = torch.tensor(
                target_point,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # Add batch dimension

            # Generate path using neural network
            with torch.no_grad():
                try:
                    path_tensor = self.model(target_tensor)
                except Exception as e:
                    self.log_error(f"Neural network inference failed: {str(e)}")
                    raise RuntimeError("Path generation failed")

                # Validate network output
                if torch.any(torch.isnan(path_tensor)):
                    raise RuntimeError("Network produced NaN values")

                path_numpy = path_tensor.squeeze(0).cpu().numpy()

            # Post-process the path
            path_points = self._post_process_path(path_numpy)

            # Validate the generated path
            if not self.validate_path(path_points):
                raise RuntimeError("Generated path fails validation")

            return self._create_polygon_stamped(path_points)

        except Exception as e:
            self.log_error(f"Path planning failed: {str(e)}")
            raise RuntimeError(f"Path planning failed: {str(e)}")

    def _post_process_path(self,
                          raw_path: np.ndarray,
                          smooth_window: int = 5) -> List[Tuple[float, float, float]]:
        """
        Post-process the neural network output into a valid needle path.

        Args:
            raw_path: Raw output from the neural network
            smooth_window: Window size for smoothing

        Returns:
            List[Tuple[float, float, float]]: List of processed path points
        """
        # Ensure path starts at origin
        raw_path[0] = [0, 0, 0]

        # Remove any NaN or Inf values
        if np.any(np.isnan(raw_path)) or np.any(np.isinf(raw_path)):
            self.log_warn("Found NaN/Inf in path, attempting to interpolate")
            mask = np.any(np.isnan(raw_path) | np.isinf(raw_path), axis=1)
            raw_path[mask] = 0.0

        # Apply smoothing
        if len(raw_path) > 3:
            try:
                from scipy.signal import savgol_filter
                window_size = min(smooth_window, len(raw_path) // 2)
                if window_size % 2 == 0:
                    window_size += 1
                if window_size >= 3:
                    for i in range(3):
                        raw_path[:, i] = savgol_filter(
                            raw_path[:, i],
                            window_size,
                            3  # polynomial order
                        )
            except Exception as e:
                self.log_warn(f"Smoothing failed: {str(e)}")

        # Ensure proper spacing
        from ..utils.path_utils import interpolate_path
        points = [(p[0], p[1], p[2]) for p in raw_path]
        return interpolate_path(points, self._parameters['output_points'])


def main():
    import rclpy
    from rclpy.node import Node

    class TestNode(Node):
        def __init__(self):
            super().__init__('test_neural_network_planner')

    rclpy.init()
    node = TestNode()

    try:
        planner = NeuralNetworkPlanner(node)
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