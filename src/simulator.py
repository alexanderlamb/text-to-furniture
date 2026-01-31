"""
Physics simulator for furniture stability testing.

Uses PyBullet to simulate furniture under gravity and measure stability metrics.
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from furniture import FurnitureDesign
from urdf_generator import save_urdf_temp


@dataclass
class SimulationResult:
    """Results from a physics simulation run."""
    # Basic pass/fail
    stable: bool
    
    # Position metrics
    initial_position: np.ndarray
    final_position: np.ndarray
    position_change: float  # meters
    
    # Rotation metrics
    initial_rotation: np.ndarray  # quaternion
    final_rotation: np.ndarray
    rotation_change: float  # radians
    
    # Height metrics
    initial_height: float
    final_height: float
    height_drop: float
    
    # Timing
    simulation_time: float  # seconds of sim time
    wall_time: float  # actual seconds elapsed
    
    # Additional metrics
    fell_over: bool
    touched_ground: bool  # Did something other than legs touch ground?
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "stable": self.stable,
            "position_change": self.position_change,
            "rotation_change": self.rotation_change,
            "height_drop": self.height_drop,
            "fell_over": self.fell_over,
            "touched_ground": self.touched_ground,
            "simulation_time": self.simulation_time,
            "wall_time": self.wall_time,
        }


class FurnitureSimulator:
    """
    PyBullet-based simulator for furniture stability testing.
    """
    
    def __init__(self, gui: bool = False, gravity: float = -9.81):
        """
        Initialize the simulator.
        
        Args:
            gui: If True, show visual GUI (slower but useful for debugging)
            gravity: Gravity acceleration (m/s²), default Earth gravity
        """
        self.gui = gui
        self.gravity = gravity
        self.physics_client = None
        self.furniture_id = None
        self.plane_id = None
        
    def _connect(self):
        """Connect to physics server."""
        if self.physics_client is not None:
            return
        
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            # Set camera position for better viewing
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0.5, 0.3]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up physics parameters
        p.setGravity(0, 0, self.gravity)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240,
            numSubSteps=4,
        )
        
        # Add ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
    
    def _disconnect(self):
        """Disconnect from physics server."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
    
    def _reset(self):
        """Reset simulation state."""
        if self.furniture_id is not None:
            p.removeBody(self.furniture_id)
            self.furniture_id = None
    
    def _load_furniture(self, urdf_path: str, start_height: float = 0.01) -> int:
        """
        Load furniture URDF into simulation.
        
        Args:
            urdf_path: Path to URDF file
            start_height: Initial height offset (meters) to prevent ground clipping
        
        Returns:
            PyBullet body ID
        """
        self.furniture_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, start_height],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
        )
        return self.furniture_id
    
    def _get_furniture_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current position and orientation of furniture."""
        pos, orn = p.getBasePositionAndOrientation(self.furniture_id)
        return np.array(pos), np.array(orn)
    
    def _quaternion_angle(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute angle between two quaternions."""
        # Dot product of quaternions
        dot = np.clip(np.abs(np.dot(q1, q2)), 0, 1)
        # Angle in radians
        return 2 * np.arccos(dot)
    
    def simulate(self, design: FurnitureDesign, 
                 duration: float = 3.0,
                 stability_threshold: float = 0.05,
                 rotation_threshold: float = 0.1) -> SimulationResult:
        """
        Simulate furniture and test stability.
        
        Args:
            design: FurnitureDesign to test
            duration: Simulation duration in seconds
            stability_threshold: Max position change (m) to be considered stable
            rotation_threshold: Max rotation change (rad) to be considered stable
        
        Returns:
            SimulationResult with stability metrics
        """
        start_wall_time = time.time()
        
        # Connect if needed
        self._connect()
        self._reset()
        
        # Generate and load URDF
        urdf_path = save_urdf_temp(design)
        try:
            self._load_furniture(urdf_path)
        finally:
            # Clean up temp file
            os.unlink(urdf_path)
        
        # Get initial state
        initial_pos, initial_orn = self._get_furniture_state()
        initial_height = initial_pos[2]
        
        # Run simulation
        steps = int(duration * 240)  # 240 Hz simulation
        for step in range(steps):
            p.stepSimulation()
            
            if self.gui:
                time.sleep(1/240)  # Real-time for visualization
            
            # Early exit if furniture falls below ground
            current_pos, _ = self._get_furniture_state()
            if current_pos[2] < -0.5:  # Fell through ground somehow
                break
        
        # Get final state
        final_pos, final_orn = self._get_furniture_state()
        
        # Compute metrics
        position_change = np.linalg.norm(final_pos[:2] - initial_pos[:2])  # XY only
        rotation_change = self._quaternion_angle(initial_orn, final_orn)
        height_drop = initial_height - final_pos[2]
        
        # Determine if fell over (large rotation)
        fell_over = rotation_change > 0.5  # ~30 degrees
        
        # Check if something touched ground that shouldn't
        # (simplified - just check if height dropped significantly)
        touched_ground = height_drop > 0.1
        
        # Overall stability
        stable = (
            position_change < stability_threshold and
            rotation_change < rotation_threshold and
            not fell_over and
            final_pos[2] > 0  # Still above ground
        )
        
        wall_time = time.time() - start_wall_time
        
        return SimulationResult(
            stable=stable,
            initial_position=initial_pos,
            final_position=final_pos,
            position_change=position_change,
            initial_rotation=initial_orn,
            final_rotation=final_orn,
            rotation_change=rotation_change,
            initial_height=initial_height,
            final_height=final_pos[2],
            height_drop=height_drop,
            simulation_time=duration,
            wall_time=wall_time,
            fell_over=fell_over,
            touched_ground=touched_ground,
        )
    
    def simulate_batch(self, designs: list, 
                       duration: float = 3.0) -> list:
        """
        Simulate multiple designs efficiently.
        
        Args:
            designs: List of FurnitureDesign objects
            duration: Simulation duration per design
        
        Returns:
            List of SimulationResult objects
        """
        results = []
        for design in designs:
            result = self.simulate(design, duration)
            results.append(result)
        return results
    
    def close(self):
        """Clean up simulator resources."""
        self._disconnect()


def quick_stability_test(design: FurnitureDesign, 
                         duration: float = 3.0,
                         gui: bool = False) -> SimulationResult:
    """
    Quick one-off stability test for a single design.
    
    Args:
        design: FurnitureDesign to test
        duration: Simulation duration
        gui: Show visual simulation
    
    Returns:
        SimulationResult
    """
    sim = FurnitureSimulator(gui=gui)
    try:
        result = sim.simulate(design, duration)
        return result
    finally:
        sim.close()


if __name__ == "__main__":
    from templates import create_simple_table
    from genome import create_table_genome, create_shelf_genome
    
    print("Testing physics simulation...")
    print("="*50)
    
    # Test with simple table
    print("\n1. Testing simple table...")
    table = create_simple_table()
    result = quick_stability_test(table, duration=2.0, gui=False)
    
    print(f"   Stable: {result.stable}")
    print(f"   Position change: {result.position_change:.4f} m")
    print(f"   Rotation change: {np.degrees(result.rotation_change):.2f} degrees")
    print(f"   Height drop: {result.height_drop:.4f} m")
    print(f"   Fell over: {result.fell_over}")
    print(f"   Wall time: {result.wall_time:.2f} s")
    
    # Test with random genome tables
    print("\n2. Testing random genome tables...")
    for i in range(5):
        genome = create_table_genome()
        design = genome.to_furniture_design(f"random_table_{i}")
        result = quick_stability_test(design, duration=2.0)
        status = "✓ STABLE" if result.stable else "✗ UNSTABLE"
        print(f"   Table {i}: {status} (rot: {np.degrees(result.rotation_change):.1f}°, "
              f"drop: {result.height_drop:.3f}m)")
    
    # Test with random shelves
    print("\n3. Testing random genome shelves...")
    for i in range(5):
        genome = create_shelf_genome()
        design = genome.to_furniture_design(f"random_shelf_{i}")
        result = quick_stability_test(design, duration=2.0)
        status = "✓ STABLE" if result.stable else "✗ UNSTABLE"
        print(f"   Shelf {i}: {status} (rot: {np.degrees(result.rotation_change):.1f}°, "
              f"drop: {result.height_drop:.3f}m)")
    
    print("\n" + "="*50)
    print("Simulation testing complete!")
