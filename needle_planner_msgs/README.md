# Needle Planner Messages

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Message Definitions](#message-definitions)
4. [Dependencies](#dependencies)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Message Details](#message-details)
8. [Examples](#examples)
9. [Development Guide](#development-guide)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

## Overview

The Needle Planner Messages package provides custom message and service definitions for needle path planning operations. These definitions establish a standardized interface for communication between needle planning components and other ROS2 nodes.

## Package Structure

```plaintext
needle_planner_msgs/
├── CMakeLists.txt
├── package.xml
└── srv/
    └── NeedlePlan.srv
```

## Message Definitions

### Service Definitions

#### NeedlePlan.srv

```plaintext
# Request
geometry_msgs/Point target          # Target point for needle insertion
string planner_type ""              # Optional: Specific planner to use

---
# Response
geometry_msgs/PolygonStamped plan   # Planned path as series of points
bool success                        # Whether planning was successful
string message                      # Additional information or error message
```

## Dependencies

### Required Packages

```xml
<buildtool_depend>ament_cmake</buildtool_depend>
<buildtool_depend>rosidl_default_generators</buildtool_depend>

<depend>geometry_msgs</depend>
<depend>std_msgs</depend>

<exec_depend>rosidl_default_runtime</exec_depend>
```

## Installation

1. Build the package:

    ```bash
    colcon build --packages-select needle_planner_msgs
    ```

2. Source setup:

    ```bash
    source install/setup.bash
    ```

## Usage

### Service Interface

```bash
# View service interface
ros2 interface show needle_planner_msgs/srv/NeedlePlan

# Call service from command line
ros2 service call /needle_planner needle_planner_msgs/srv/NeedlePlan \
    "{target: {x: 0.0, y: 0.0, z: 100.0}}"
```

### Programming Interface

```python
from needle_planner_msgs.srv import NeedlePlan
from geometry_msgs.msg import Point

class PlanningClient:
    def __init__(self):
        self.cli = self.create_client(NeedlePlan, 'needle_planner')

    async def plan_path(self, x: float, y: float, z: float):
        request = NeedlePlan.Request()
        request.target = Point(x=x, y=y, z=z)
        response = await self.cli.call_async(request)
        return response
```

## Message Details

### NeedlePlan Service

#### Request Fields

1. `target` (geometry_msgs/Point)
   - `x`: X-coordinate (mm)
   - `y`: Y-coordinate (mm)
   - `z`: Z-coordinate (mm)

2. `planner_type` (string)
   - Optional field
   - Valid values:
     - "linear_regression"
     - "neural_network"
     - "geometric"
     - "hybrid"
     - "sampling"

#### Response Fields

1. `plan` (geometry_msgs/PolygonStamped)
   - `header`: Standard ROS header
   - `polygon`: Series of points forming path
     - Points are in order from start to target
     - First point is at origin (0,0,0)
     - Last point is at target

2. `success` (bool)
   - True if planning succeeded
   - False if planning failed

3. `message` (string)
   - Success/error description
   - Additional information

## Examples

### Python Example

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from needle_planner_msgs.srv import NeedlePlan
from geometry_msgs.msg import Point

class PlanningClient(Node):
    def __init__(self):
        super().__init__('planning_client')
        self.cli = self.create_client(NeedlePlan, 'needle_planner')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        self.req = NeedlePlan.Request()

    def send_request(self, x, y, z):
        self.req.target = Point(x=x, y=y, z=z)
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    client = PlanningClient()
    response = client.send_request(0.0, 0.0, 100.0)
    print(f"Success: {response.success}")
    print(f"Message: {response.message}")
    print(f"Path points: {len(response.plan.polygon.points)}")
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### C++ Example

```cpp
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <needle_planner_msgs/srv/needle_plan.hpp>

class PlanningClient : public rclcpp::Node
{
public:
    PlanningClient() : Node("planning_client")
    {
        client_ = create_client<needle_planner_msgs::srv::NeedlePlan>(
            "needle_planner");
    }

    void send_request(double x, double y, double z)
    {
        auto request = std::make_shared<needle_planner_msgs::srv::NeedlePlan::Request>();
        request->target.x = x;
        request->target.y = y;
        request->target.z = z;

        auto result = client_->async_send_request(request);
        // Wait for result...
    }

private:
    rclcpp::Client<needle_planner_msgs::srv::NeedlePlan>::SharedPtr client_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlanningClient>();
    node->send_request(0.0, 0.0, 100.0);
    rclcpp::shutdown();
    return 0;
}
```

## Development Guide

### Adding New Messages

1. Create message definition:

    ```msg
    # my_message.msg
    std_msgs/Header header
    # ... other fields
    ```

2. Update CMakeLists.txt:

    ```cmake
    rosidl_generate_interfaces(${PROJECT_NAME}
      "msg/MyMessage.msg"
      DEPENDENCIES geometry_msgs std_msgs
    )
    ```

3. Rebuild package:

    ```bash
    colcon build --packages-select needle_planner_msgs
    ```

### Message Evolution

- Maintain backward compatibility
- Use optional fields for extensions
- Document changes thoroughly
- Update major version for breaking changes

## Testing

### Interface Tests

```bash
# Test service availability
ros2 service list | grep needle_planner

# Test interface definition
ros2 interface show needle_planner_msgs/srv/NeedlePlan

# Test service call
ros2 service call /needle_planner needle_planner_msgs/srv/NeedlePlan \
    "{target: {x: 0.0, y: 0.0, z: 100.0}}"
```

### Message Generation Tests

```bash
# Verify message generation
ros2 interface list | grep needle_planner_msgs

# Check message dependencies
ros2 interface dependencies needle_planner_msgs/srv/NeedlePlan
```

## Troubleshooting

### Common Issues

1. Message Generation Errors

    ```plaintext
    Problem: Failed to generate message interfaces
    Solution:
    - Check message syntax
    - Verify dependencies
    - Clean build directory
    ```

2. Service Communication Issues

    ```plaintext
    Problem: Service calls fail
    Solution:
    - Check node configuration
    - Verify message types
    - Monitor ROS2 network
    ```

3. Version Compatibility

    ```plaintext
    Problem: Interface version mismatch
    Solution:
    - Update all packages
    - Check API compatibility
    - Rebuild workspace
    ```

### Debugging Tools

```bash
# Check message interfaces
ros2 interface list | grep needle_planner_msgs

# Monitor service calls
ros2 service type /needle_planner

# View node information
ros2 node info /needle_planning_node

# Check type information
ros2 interface proto needle_planner_msgs/srv/NeedlePlan
```
