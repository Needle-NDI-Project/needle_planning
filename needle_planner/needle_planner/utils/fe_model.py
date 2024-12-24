"""
Finite Element Model for needle deflection analysis.
"""

import enum
from typing import List, Tuple, Optional
import numpy as np
from PyNite import FEModel3D, Node3D, Element3D

# Physical parameters (all units in mm, N, MPa)
ELASTIC_MODULUS = 35e3  # Elastic Modulus of Needle (Nitinol, MPa)
POISSON_RATIO = 0.33
SHEAR_MODULUS = ELASTIC_MODULUS / (2 * (1 + POISSON_RATIO))
OUTER_DIAMETER = 1.270  # mm
INNER_DIAMETER = 0.838  # mm
AREA = np.pi * (OUTER_DIAMETER/2)**2 - np.pi * (INNER_DIAMETER/2)**2
J = np.pi/32 * (OUTER_DIAMETER**4 - INNER_DIAMETER**4)  # Polar Moment of Inertia
I = np.pi/64 * (OUTER_DIAMETER**4 - INNER_DIAMETER**4)  # Moment of Inertia
GUIDE_GAP = 5  # mm
GUIDE_THICKNESS = 10  # mm


class ExitCode(enum.Enum):
    """Exit codes for FE analysis."""
    SUCCESS = 0
    INVALID_INPUT = 1
    CONVERGENCE_ERROR = 2
    STABILITY_ERROR = 3
    UNKNOWN_ERROR = 4


def run_model(
    tip_pos: np.ndarray,
    y_rxn: np.ndarray,
    y_disp: np.ndarray,
    stiffness_vec: np.ndarray,
    pos_vec: np.ndarray,
    tip_path: List[float],
    guide_pos_vec: np.ndarray,
    guide_pos_ins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Run finite element analysis for needle deflection.

    Args:
        tip_pos: Target tip position [x, y, z]
        y_rxn: Reaction forces in y-direction
        y_disp: Current displacement in y-direction
        stiffness_vec: Vector of tissue stiffness values
        pos_vec: Vector of position values where stiffness changes
        tip_path: Historical path of tip movement
        guide_pos_vec: Guide positions
        guide_pos_ins: Guide insertion positions

    Returns:
        Tuple containing:
        - Updated y displacement
        - Updated reaction forces
        - Updated tip path

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If FE analysis fails
    """
    try:
        # Input validation
        _validate_inputs(tip_pos, y_rxn, y_disp, stiffness_vec, pos_vec)

        # Initialize FE model
        model = _initialize_model()

        # Setup nodes and elements
        L = 140  # mm (maximum needle length)
        offset = tip_pos[2] - L
        num_nodes = round(L/3) + 1
        spacing = L/round(L/3)

        # Add nodes and loads
        _setup_nodes_and_loads(model, num_nodes, spacing, offset, y_rxn,
                             guide_pos_vec, guide_pos_ins, tip_pos)

        # Add elements
        _add_elements(model, num_nodes)

        # Analyze model
        result = _analyze_model(model)

        if result != ExitCode.SUCCESS:
            raise RuntimeError(f"FE analysis failed with code: {result}")

        # Extract results
        new_y_disp, new_y_rxn = _extract_results(model, num_nodes)

        # Update tip path history
        tip_path.append(new_y_disp[-1])

        return new_y_disp, new_y_rxn, tip_path

    except Exception as e:
        raise RuntimeError(f"FE analysis failed: {str(e)}")


def _validate_inputs(tip_pos: np.ndarray,
                    y_rxn: np.ndarray,
                    y_disp: np.ndarray,
                    stiffness_vec: np.ndarray,
                    pos_vec: np.ndarray) -> None:
    """Validate input parameters."""
    if not all(isinstance(arr, np.ndarray) for arr in
              [tip_pos, y_rxn, y_disp, stiffness_vec, pos_vec]):
        raise ValueError("All inputs must be numpy arrays")

    if tip_pos.shape != (3,):
        raise ValueError("tip_pos must be a 3D point")

    if len(y_rxn) != len(y_disp):
        raise ValueError("y_rxn and y_disp must have same length")

    if len(stiffness_vec) != len(pos_vec):
        raise ValueError("stiffness_vec and pos_vec must have same length")

    if np.any(np.isnan(tip_pos)) or np.any(np.isnan(y_rxn)) or \
       np.any(np.isnan(y_disp)) or np.any(np.isnan(stiffness_vec)) or \
       np.any(np.isnan(pos_vec)):
        raise ValueError("Input arrays contain NaN values")


def _initialize_model() -> FEModel3D:
    """Initialize the FE model with material properties."""
    model = FEModel3D()

    # Add material
    model.add_material('Nitinol', ELASTIC_MODULUS, SHEAR_MODULUS,
                      POISSON_RATIO, 6.45e-9)

    return model


def _setup_nodes_and_loads(model: FEModel3D,
                          num_nodes: int,
                          spacing: float,
                          offset: float,
                          y_rxn: np.ndarray,
                          guide_pos_vec: np.ndarray,
                          guide_pos_ins: np.ndarray,
                          tip_pos: np.ndarray) -> None:
    """Setup nodes and apply loads/boundary conditions."""
    def get_guide_pos(z_pos: float) -> float:
        """Get guide position for given z position."""
        for i, ins_pos in enumerate(guide_pos_ins):
            if z_pos < ins_pos:
                return guide_pos_vec[i]
        return guide_pos_vec[-1]

    for i in range(num_nodes):
        node_z = i * spacing + offset
        node_name = f'N{i + 1}'

        # Add node
        model.add_node(node_name, 0, 0, node_z)

        # Apply reaction forces
        if i > 0:  # Skip first node
            model.add_node_load(node_name, 'Fy', y_rxn[i-1])

        # Apply boundary conditions
        if node_z < -GUIDE_GAP:
            # Node in guide - fixed displacement
            model.def_support(node_name, True, True, True, True, True, True)
            model.def_node_disp(node_name, 'dy', get_guide_pos(node_z))
        elif -GUIDE_GAP <= node_z <= 0:
            # Node in transition region - partial constraint
            model.def_support(node_name, True, False, True, True, True, True)
        else:
            # Node in tissue - elastic support
            spring_stiffness = _get_tissue_stiffness(node_z)
            model.def_support_spring(node_name, 'dy', spring_stiffness)

    # Apply tip load
    tip_angle = np.arctan2(tip_pos[1], tip_pos[0])
    tip_force = _compute_tip_force(tip_angle)
    model.add_node_load(f'N{num_nodes}', 'Fy', -tip_force)


def _get_tissue_stiffness(z_pos: float) -> float:
    """Get tissue stiffness at given position."""
    # Simplified linear interpolation between stiffness values
    # In practice, this would use the stiffness_vec and pos_vec
    base_stiffness = 0.05  # N/mm
    stiffness_gradient = 0.0002  # N/mm²
    return base_stiffness + stiffness_gradient * z_pos


def _compute_tip_force(tip_angle: float) -> float:
    """Compute force at needle tip."""
    # Simplified force model based on tip angle
    base_force = 0.5  # N
    return base_force * np.cos(tip_angle)


def _add_elements(model: FEModel3D, num_nodes: int) -> None:
    """Add beam elements to the model."""
    # Add elements connecting nodes
    for i in range(1, num_nodes):
        model.add_element(
            f'E{i}',
            f'N{i}',
            f'N{i+1}',
            'Nitinol',
            I, I, J, AREA
        )


def _analyze_model(model: FEModel3D) -> ExitCode:
    """Analyze the FE model."""
    try:
        model.analyze(check_statics=True, check_stability=True)
        return ExitCode.SUCCESS
    except Exception as e:
        if "convergence" in str(e).lower():
            return ExitCode.CONVERGENCE_ERROR
        elif "stability" in str(e).lower():
            return ExitCode.STABILITY_ERROR
        else:
            return ExitCode.UNKNOWN_ERROR


def _extract_results(model: FEModel3D,
                    num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract displacement and reaction force results."""
    y_disp = np.zeros(num_nodes)
    y_rxn = np.zeros(num_nodes)

    for i in range(num_nodes):
        node_name = f'N{i+1}'
        y_disp[i] = model.Nodes[node_name].DY['Combo 1']
        y_rxn[i] = model.Nodes[node_name].RxnFY['Combo 1']

    return y_disp, y_rxn