# Needle Planning

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-green)](https://docs.ros.org/en/humble/)

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Packages](#packages)
6. [Usage](#usage)
7. [Technical Details](#technical-details)
8. [Development Guides](#development-guides)

## Overview

The **Needle Planning** repository provides a comprehensive suite of ROS2 packages for surgical needle path planning. It includes various planning approaches from geometric methods to machine learning-based solutions, offering flexibility and robustness for different surgical scenarios.

### Features

- Multiple planning approaches:
  - Geometric planning with circular arc approximation
  - Machine learning-based planning (linear regression and neural networks)
  - Hybrid planning combining geometric and ML approaches
  - Sampling-based planning with RRT-like exploration
- Real-time path generation and validation
- Integration with NDI tracking systems
- Comprehensive testing suite
- Modular architecture for easy extension

## Repository Structure

```plaintext
needle_planning/
├── needle_planner/             # Core planning package
│   ├── config/                 # Configuration files
│   ├── launch/                 # Launch files
│   ├── needle_planner/         # Python package
│   │   ├── planners/           # Planning implementations
│   │   └── utils/              # Utility functions
│   └── test/                   # Unit tests
├── needle_planner_msgs/        # Message definitions
│   └── srv/                    # Service definitions
└── README.md                   # This file
```

## Prerequisites

### Required Software

- Ubuntu 22.04 or later
- ROS2 Humble or later
- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, scikit-learn
- PyNiteFEA

### ROS2 Dependencies

```xml
<depend>rclpy</depend>
<depend>geometry_msgs</depend>
<depend>tf2_ros</depend>
<depend>tf2_geometry_msgs</depend>
```

## Installation

1. Create a new workspace:

    ```bash
    mkdir -p needle_ws/src
    cd needle_ws/src
    ```

2. Clone the repository:

    ```bash
    git clone https://github.com/Needle-NDI-Project/needle_planning.git
    ```

3. Install Python dependencies:

    ```bash
    pip3 install -r needle_planner/requirements.txt
    ```

4. Install ROS dependencies:

    ```bash
    rosdep install --from-paths src --ignore-src -r -y
    ```

5. Build the workspace:

    ```bash
    cd ..
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
    ```

6. Source the workspace:

    ```bash
    source install/setup.bash
    ```

## Packages

### needle_planner

Core package implementing various planning strategies:

- Geometric planning
- Machine learning-based planning
- Hybrid planning
- Sampling-based planning

See [needle_planner/README.md](needle_planner/README.md) for details.

### needle_planner_msgs

Message and service definitions for needle planning:

- Path planning service
- Custom message types

See [needle_planner_msgs/README.md](needle_planner_msgs/README.md) for details.

## Usage

### Basic Usage

1. Launch the planning system:

    ```bash
    ros2 launch needle_planner needle_planner.launch.py
    ```

2. Request a path plan:

    ```bash
    ros2 service call /needle_planner needle_planner_msgs/srv/NeedlePlan \
        "{target: {x: 0.0, y: 0.0, z: 100.0}}"
    ```

### Advanced Usage

1. Specify planner type:

    ```bash
    ros2 launch needle_planner needle_planner.launch.py planner_type:=geometric
    ```

2. Use custom configuration:

    ```bash
    ros2 launch needle_planner needle_planner.launch.py \
        config_file:=path/to/config.yaml
    ```

## Technical Details

### Planning Approaches

1. **Geometric Planning**
   - Circular arc approximation
   - Curvature constraints
   - Workspace validation

2. **Machine Learning**
   - Linear regression with PCA
   - Neural network path generation
   - Data-driven corrections

3. **Hybrid Planning**
   - Combined geometric and ML
   - FE-based refinement
   - Adaptive corrections

4. **Sampling-based Planning**
   - RRT-like exploration
   - Dynamic replanning
   - Constraint satisfaction

### Performance Considerations

- Update rates: Up to 10Hz
- Planning latency: <100ms typical
- Memory usage: <500MB
- CPU usage: <30% on modern processors

### Safety Features

- Workspace boundary checking
- Curvature constraint enforcement
- Path validation
- Real-time monitoring

## Development Guides

### Adding New Planners

1. Create new planner class:

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
