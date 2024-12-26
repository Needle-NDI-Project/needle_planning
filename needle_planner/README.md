# Needle Planner

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-green)](https://docs.ros.org/en/humble/)

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Mathematical Framework](#mathematical-framework)
4. [Planning Approaches](#planning-approaches)
5. [Dependencies](#dependencies)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Usage](#usage)
9. [API Reference](#api-reference)
10. [Development Guide](#development-guide)
11. [Troubleshooting](#troubleshooting)

## Overview

The Needle Planner package provides comprehensive solutions for surgical needle path planning using various approaches including geometric, machine learning, hybrid, and sampling-based methods. The package is designed for real-time operation in surgical environments, with emphasis on accuracy, reliability, and computational efficiency.

### Key Features

- Multiple planning strategies with real-time performance
- Path validation and optimization
- Integration with tracking systems
- Extensive configuration options
- Comprehensive testing suite
- Thread-safe implementation
- Real-time performance monitoring

## Package Structure

```plaintext
needle_planner/
├── config/
│   ├── default_params.yaml         # Default parameters
│   └── planner_configs/            # Planner-specific configs
│       ├── geometric.yaml
│       ├── hybrid.yaml
│       ├── linear_regression.yaml
│       └── neural_network.yaml
├── launch/
│   └── needle_planner.launch.py    # Launch file
└── needle_planner/
    ├── planners/                   # Planning implementations
    │   ├── base_planner.py         # Abstract base class
    │   ├── geometric_planner.py    # Geometric approach
    │   ├── hybrid_planner.py       # Hybrid approach
    │   ├── linear_regression_planner.py
    │   ├── neural_network_planner.py
    │   └── sampling_planner.py     # RRT-based planning
    └── utils/                      # Utility functions
        ├── fe_model.py             # FE analysis
        └── path_utils.py           # Path manipulation
```

## Mathematical Framework

### Coordinate System

The planner operates in a 3D Cartesian coordinate system where:

- Origin (0,0,0): Needle insertion point
- Z-axis: Insertion direction
- X-Y plane: Transverse plane

### Path Representation

A path is represented as a series of points P(t) = [x(t), y(t), z(t)] where t ∈ [0,1]:

```math
P(t) = \begin{bmatrix}
x(t) \\
y(t) \\
z(t)
\end{bmatrix}, \quad t \in [0,1]
```

### Curvature Constraints

Needle curvature κ is constrained by material properties:

```math
κ(t) = \frac{\|\dot{P}(t) × \ddot{P}(t)\|}{\|\dot{P}(t)\|^3} ≤ κ_{\max}
```

where κ_max is the maximum allowable curvature.

### Path Optimization

The path optimization objective function combines multiple terms:

```math
J(P) = w_1J_{\text{length}}(P) + w_2J_{\text{curvature}}(P) + w_3J_{\text{tissue}}(P)
```

where:

- J_length: Path length term
- J_curvature: Curvature smoothness term
- J_tissue: Tissue interaction term
- w_i: Weighting factors

## Planning Approaches

### 1. Geometric Planning

The geometric planner uses circular arc approximation based on the following principles:

#### Arc Generation

Given target point T = [x_t, y_t, z_t], the planner computes:

1. Chord length:

    ```math
    L = \|T\|
    ```

2. Deviation angle:

    ```math
    θ = \arccos\left(\frac{z_t}{L}\right)
    ```

3. Arc parameters:

    ```math
    R = \frac{L}{2\sin(θ/2)}  \text{ (radius)}
    κ = \frac{1}{R}  \text{ (curvature)}
    ```

#### Implementation

```python
def compute_circular_arc(self, target: Point) -> List[Tuple[float, float, float]]:
    """Compute circular arc path."""
    chord_length = np.linalg.norm([target.x, target.y, target.z])
    theta = np.arccos(target.z / chord_length)

    # Generate arc points
    t = np.linspace(0, theta, self._parameters['num_points'])
    R = chord_length / (2 * np.sin(theta/2))

    points = []
    for angle in t:
        x = R * (1 - np.cos(angle))
        z = R * np.sin(angle)
        points.append((x, 0, z))

    return points
```

### 2. Machine Learning Planning

#### Linear Regression with PCA

1. Dimensionality Reduction:

    ```math
    X_{\text{reduced}} = \text{PCA}(X_{\text{original}}, n_{\text{components}})
    ```

2. Feature Generation:

    ```math
    Φ(x) = [1, x, x^2, ..., x^p]  \text{ (polynomial features)}
    ```

3. Path Prediction:

    ```math
    y = Φ(X_{\text{reduced}})β + ε
    ```

#### Neural Network Approach

Architecture:

```python
class NeedlePathNN(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[32, 64, 128]):
        super().__init__()
        layers = []
        prev_size = input_dim

        for size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size)
            ])
            prev_size = size

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_layers[-1], output_points * 3),
            nn.Tanh()
        )
```

### 3. Hybrid Planning

The hybrid planner combines geometric and ML approaches:

1. Initial path from geometric planner
2. ML-based correction:

    ```math
    P_{\text{corrected}}(t) = P_{\text{geometric}}(t) + α\Delta P_{\text{ML}}(t)
    ```

3. FE refinement using beam theory:

    ```math
    EI\frac{d^4w}{dx^4} + kw = q(x)
    ```

    where:

    - E: Young's modulus
    - I: Area moment of inertia
    - k: Tissue stiffness
    - w: Deflection
    - q(x): Distributed load

### 4. Sampling-based Planning

RRT-like algorithm with the following modifications:

1. Node Selection:

    ```math
    J_{\text{node}} = w_1\|x - x_{\text{nearest}}\| + w_2κ_{\text{path}}
    ```

2. Extension Step:

    ```python
    def extend_towards(self, from_pos: np.ndarray, to_pos: np.ndarray) -> Optional[np.ndarray]:
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        step = min(self._parameters['step_size'], distance)
        new_direction = direction / distance
        new_pos = from_pos + new_direction * step

        if self._check_extension_valid(from_pos, new_pos):
            return new_pos
        return None
    ```

## Dependencies

### Required Packages

```xml
<depend>rclpy</depend>
<depend>geometry_msgs</depend>
<depend>needle_planner_msgs</depend>
<depend>tf2_ros</depend>
<depend>tf2_geometry_msgs</depend>
```

### Python Dependencies

```requirements
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=0.24.0
torch>=1.9.0
PyNiteFEA>=0.0.8
```

## Installation

1. Create workspace:

    ```bash
    mkdir -p needle_ws/src
    cd needle_ws/src
    git clone https://github.com/Needle-NDI-Project/needle_planning.git
    ```

2. Install dependencies:

    ```bash
    pip3 install -r needle_planner/requirements.txt
    cd ..
    rosdep install --from-paths src --ignore-src -r -y
    ```

3. Build package:

    ```bash
    colcon build --packages-select needle_planner
    source install/setup.bash
    ```

## Configuration

### Default Parameters

```yaml
default_planner: 'linear_regression'

linear_regression:
  model:
    path: 'models/linear_regression_model.pkl'
    polynomial_degree: 2
  path_generation:
    num_points: 50
    spacing: 0.5  # mm

neural_network:
  model:
    architecture:
      input_dim: 3
      hidden_layers: [32, 64, 128]
      output_points: 20
  inference:
    device: 'cpu'
    batch_size: 1
```

### Validation Parameters

```yaml
validation:
  max_curvature: 0.01  # 1/mm
  min_spacing: 0.5     # mm
  max_depth: 120.0     # mm
  max_lateral: 50.0    # mm
```

## Usage

### Basic Usage

1. Launch planner:

    ```bash
    ros2 launch needle_planner needle_planner.launch.py
    ```

2. Request path:

    ```bash
    ros2 service call /needle_planner needle_planner_msgs/srv/NeedlePlan \
        "{target: {x: 0.0, y: 0.0, z: 100.0}}"
    ```

### Advanced Usage

1. Specify planner:

    ```bash
    ros2 launch needle_planner needle_planner.launch.py planner_type:=geometric
    ```

2. Custom configuration:

    ```bash
    ros2 launch needle_planner needle_planner.launch.py \
        config_file:=path/to/config.yaml
    ```

## API Reference

### Base Planner Interface

```python
class BasePlanner(ABC):
    @abstractmethod
    def compute_plan(self, target: Point) -> PolygonStamped:
        """Compute needle path plan."""
        pass

    def validate_target(self, target: Point) -> bool:
        """Validate target point."""
        pass

    def validate_path(self, points: List[Tuple[float, float, float]]) -> bool:
        """Validate generated path."""
        pass
```

### Planning Node

```python
class NeedlePlanningNode(Node):
    def __init__(self):
        """Initialize planning node."""
        pass

    def _plan_callback(self, request, response):
        """Handle planning service requests."""
        pass
```

### Utility Functions

```python
def interpolate_path(points: List[Tuple[float, float, float]],
                    num_points: int,
                    method: str = 'cubic') -> List[Tuple[float, float, float]]:
    """Interpolate path with specified number of points."""
    pass

def compute_curvature(points: List[Tuple[float, float, float]]) -> np.ndarray:
    """Compute discrete curvature along path."""
    pass

def validate_path(points: List[Tuple[float, float, float]],
                 max_curvature: float = 0.01,
                 min_spacing: float = 0.5) -> bool:
    """Validate path constraints."""
    pass
```

## Development Guide

### Adding New Planners

1. Create planner class:

    ```python
    from needle_planner.planners import BasePlanner

    class MyPlanner(BasePlanner):
        def compute_plan(self, target):
            # Implementation
            pass
    ```

2. Register planner:

    ```python
    # In planner_node.py
    self.planners['my_planner'] = MyPlanner(self)
    ```

## Troubleshooting

### Common Issues

1. Planning Failures

    ```plaintext
    Problem: Path planning fails with curvature constraints
    Solution:
    - Check target feasibility
    - Verify curvature limits
    - Review workspace bounds
    ```

2. Performance Issues

    ```plaintext
    Problem: Slow planning response
    Solution:
    - Adjust planning parameters
    - Check system resources
    - Monitor memory usage
    ```

### Debugging Tools

```bash
# Enable debug logging
ros2 run needle_planner planner_node --ros-args --log-level debug

# Monitor planning service
ros2 service type /needle_planner

# Check node status
ros2 node info /needle_planning_node
```

### Configuration Validation

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('config/default_params.yaml'))"

# List parameters
ros2 param list /needle_planning_node
```
