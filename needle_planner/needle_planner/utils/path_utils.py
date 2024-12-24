"""
Utility functions for path manipulation and validation.
"""

from typing import List, Tuple, Optional
import numpy as np
from geometry_msgs.msg import Point, Point32
from scipy.interpolate import CubicSpline


def interpolate_path(points: List[Tuple[float, float, float]],
                    num_points: int,
                    method: str = 'cubic') -> List[Tuple[float, float, float]]:
    """
    Interpolate path to desired number of points.

    Args:
        points: List of (x,y,z) points
        num_points: Desired number of points
        method: Interpolation method ('linear' or 'cubic')

    Returns:
        Interpolated path points

    Raises:
        ValueError: If invalid input or method
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points for interpolation")

    points_array = np.array(points)

    # Calculate cumulative distance along path
    diffs = np.diff(points_array, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    distances = np.concatenate(([0], np.cumsum(segment_lengths)))

    # Create parameter for interpolation
    t_new = np.linspace(0, distances[-1], num_points)

    if method == 'linear':
        # Linear interpolation for each dimension
        x = np.interp(t_new, distances, points_array[:, 0])
        y = np.interp(t_new, distances, points_array[:, 1])
        z = np.interp(t_new, distances, points_array[:, 2])

    elif method == 'cubic':
        # Cubic spline interpolation
        try:
            cs_x = CubicSpline(distances, points_array[:, 0], bc_type='natural')
            cs_y = CubicSpline(distances, points_array[:, 1], bc_type='natural')
            cs_z = CubicSpline(distances, points_array[:, 2], bc_type='natural')

            x = cs_x(t_new)
            y = cs_y(t_new)
            z = cs_z(t_new)
        except Exception:
            # Fallback to linear if cubic fails
            x = np.interp(t_new, distances, points_array[:, 0])
            y = np.interp(t_new, distances, points_array[:, 1])
            z = np.interp(t_new, distances, points_array[:, 2])

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return list(zip(x, y, z))


def validate_path(points: List[Tuple[float, float, float]],
                 max_curvature: float = 0.01,
                 min_spacing: float = 0.5) -> bool:
    """
    Validate path constraints.

    Args:
        points: List of (x,y,z) points
        max_curvature: Maximum allowable curvature (1/mm)
        min_spacing: Minimum spacing between points (mm)

    Returns:
        bool: True if path is valid, False otherwise
    """
    if len(points) < 2:
        return False

    points_array = np.array(points)

    # Check minimum spacing
    distances = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
    if np.any(distances < min_spacing):
        return False

    # Check curvature
    if len(points) > 2:
        curvature = compute_curvature(points)
        if np.any(curvature > max_curvature):
            return False

    return True


def compute_curvature(points: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Compute discrete curvature along path.

    Args:
        points: List of (x,y,z) points

    Returns:
        Array of curvature values

    Notes:
        Uses Menger curvature for discrete points
    """
    points_array = np.array(points)
    num_points = len(points_array)
    curvature = np.zeros(num_points)

    # Compute curvature for interior points
    for i in range(1, num_points - 1):
        prev_point = points_array[i-1]
        curr_point = points_array[i]
        next_point = points_array[i+1]

        # Vectors between points
        v1 = curr_point - prev_point
        v2 = next_point - curr_point

        # Compute angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical issues
        angle = np.arccos(cos_angle)

        # Approximate curvature using angle and segment length
        avg_length = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
        if avg_length > 1e-6:  # Avoid division by zero
            curvature[i] = angle / avg_length

    # Set end points curvature same as nearest computed value
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    return curvature


def smooth_path(points: List[Tuple[float, float, float]],
               window_size: int = 3,
               preserve_ends: bool = True) -> List[Tuple[float, float, float]]:
    """
    Apply smoothing to path points.

    Args:
        points: List of (x,y,z) points
        window_size: Size of smoothing window
        preserve_ends: Whether to preserve endpoint positions

    Returns:
        Smoothed path points
    """
    if len(points) < 3 or window_size < 3:
        return points

    points_array = np.array(points)

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    # Apply Savitzky-Golay filter for smoothing
    try:
        from scipy.signal import savgol_filter
        smoothed = np.zeros_like(points_array)

        # Keep endpoints unchanged if requested
        if preserve_ends:
            smoothed[0] = points_array[0]
            smoothed[-1] = points_array[-1]
            # Smooth interior points
            for i in range(3):  # For each dimension
                smoothed[1:-1, i] = savgol_filter(
                    points_array[:, i],
                    window_size,
                    3  # polynomial order
                )[1:-1]
        else:
            # Smooth all points
            for i in range(3):
                smoothed[:, i] = savgol_filter(
                    points_array[:, i],
                    window_size,
                    3
                )
    except ImportError:
        # Fallback to simple moving average if scipy not available
        kernel = np.ones(window_size) / window_size
        smoothed = np.zeros_like(points_array)

        if preserve_ends:
            smoothed[0] = points_array[0]
            smoothed[-1] = points_array[-1]
            # Smooth interior points
            for i in range(3):
                smoothed[1:-1, i] = np.convolve(
                    points_array[:, i],
                    kernel,
                    mode='valid'
                )
        else:
            # Smooth all points
            pad_size = window_size // 2
            for i in range(3):
                padded = np.pad(points_array[:, i], pad_size, mode='edge')
                smoothed[:, i] = np.convolve(padded, kernel, mode='valid')

    return [(p[0], p[1], p[2]) for p in smoothed]


def resample_path(points: List[Tuple[float, float, float]],
                 spacing: float) -> List[Tuple[float, float, float]]:
    """
    Resample path with uniform spacing.

    Args:
        points: List of (x,y,z) points
        spacing: Desired spacing between points (mm)

    Returns:
        Resampled path points
    """
    if len(points) < 2:
        return points

    points_array = np.array(points)

    # Calculate cumulative distance along path
    diffs = np.diff(points_array, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)

    # Calculate number of points needed
    num_points = max(2, int(np.ceil(total_length / spacing)))

    # Use interpolate_path with cubic interpolation
    return interpolate_path(points, num_points, method='cubic')


def convert_to_point32(point: Tuple[float, float, float]) -> Point32:
    """Convert tuple to Point32 message."""
    p = Point32()
    p.x = float(point[0])
    p.y = float(point[1])
    p.z = float(point[2])
    return p


def convert_to_point(point: Tuple[float, float, float]) -> Point:
    """Convert tuple to Point message."""
    p = Point()
    p.x = float(point[0])
    p.y = float(point[1])
    p.z = float(point[2])
    return p