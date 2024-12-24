#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from needle_planner_msgs.srv import NeedlePlan

from .planners.linear_regression_planner import LinearRegressionPlanner
from .planners.neural_network_planner import NeuralNetworkPlanner
from .planners.geometric_planner import GeometricPlanner
from .planners.hybrid_planner import HybridPlanner
from .planners.sampling_planner import SamplingPlanner

class NeedlePlanningNode(Node):
    """
    Main ROS2 node for needle path planning.

    This node provides services for different planning approaches and manages
    the lifecycle of all planner instances. It can dynamically switch between
    planning methods based on service parameters or configuration.
    """

    def __init__(self):
        """Initialize the needle planning node."""
        super().__init__('needle_planning_node')

        # Initialize planners
        self.planners = {
            'linear_regression': LinearRegressionPlanner(self),
            'neural_network': NeuralNetworkPlanner(self),
            'geometric': GeometricPlanner(self),
            'hybrid': HybridPlanner(self),
            'sampling': SamplingPlanner(self)
        }

        # Default planner
        self.default_planner = 'linear_regression'

        # Create planning services
        self._create_services()

        # Load parameters
        self._load_parameters()

        self.get_logger().info('Needle planning node initialized')

    def _create_services(self):
        """Create planning services for each planner type."""
        # Main planning service (uses default or specified planner)
        self.srv = self.create_service(
            NeedlePlan,
            'needle_planner',
            self._plan_callback
        )

        # Individual services for each planner type
        for planner_name in self.planners:
            self.create_service(
                NeedlePlan,
                f'needle_planner/{planner_name}',
                lambda req, resp, name=planner_name: self._specific_planner_callback(req, resp, name)
            )

    def _load_parameters(self):
        """Load parameters from ROS parameter server."""
        self.declare_parameter('default_planner', self.default_planner)
        self.default_planner = self.get_parameter(
            'default_planner').get_parameter_value().string_value

    def _plan_callback(self, request, response):
        """
        Callback for the main planning service.

        Args:
            request: Service request containing target point
            response: Service response for path plan

        Returns:
            Service response containing planned path
        """
        planner_name = self.default_planner

        self.get_logger().info(
            f'Planning request for target: ({request.target.x:.2f}, '
            f'{request.target.y:.2f}, {request.target.z:.2f}) '
            f'using {planner_name} planner'
        )

        try:
            planner = self.planners[planner_name]
            response.plan = planner.compute_plan(request.target)

            self.get_logger().info(
                f'Generated plan with {len(response.plan.polygon.points)} points'
            )

        except Exception as e:
            self.get_logger().error(f'Planning failed: {str(e)}')
            raise e

        return response

    def _specific_planner_callback(self, request, response, planner_name):
        """
        Callback for planner-specific services.

        Args:
            request: Service request containing target point
            response: Service response for path plan
            planner_name: Name of the specific planner to use

        Returns:
            Service response containing planned path
        """
        self.get_logger().info(
            f'Specific planning request using {planner_name} planner for '
            f'target: ({request.target.x:.2f}, {request.target.y:.2f}, '
            f'{request.target.z:.2f})'
        )

        try:
            planner = self.planners[planner_name]
            response.plan = planner.compute_plan(request.target)

            self.get_logger().info(
                f'Generated plan with {len(response.plan.polygon.points)} points'
            )

        except Exception as e:
            self.get_logger().error(f'Planning failed: {str(e)}')
            raise e

        return response

def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = NeedlePlanningNode()
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f'Node failed: {str(e)}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()