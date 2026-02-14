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
    
    # Load test metrics
    load_test_passed: bool = True  # Did the load stay on the surface?
    load_final_height: float = 0.0  # Final height of test load
    
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
            "load_test_passed": self.load_test_passed,
            "load_final_height": self.load_final_height,
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
        if self.physics_client is not None and p.getConnectionInfo(self.physics_client)["isConnected"]:
            return
        
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            if self.physics_client < 0:
                self.physics_client = p.connect(p.DIRECT)
            # Set camera position for better viewing
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0.5, 0.3],
                physicsClientId=self.physics_client
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up physics parameters
        p.setGravity(0, 0, self.gravity, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240,
            numSubSteps=4,
            physicsClientId=self.physics_client
        )
        
        # Add ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
    
    def _disconnect(self):
        """Disconnect from physics server."""
        if self.physics_client is not None:
            if p.getConnectionInfo(self.physics_client)["isConnected"]:
                p.disconnect(self.physics_client)
            self.physics_client = None
    
    def _reset(self):
        """Reset simulation state."""
        if self.furniture_id is not None:
            try:
                p.removeBody(self.furniture_id, physicsClientId=self.physics_client)
            except p.error:
                pass
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
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self.physics_client
        )
        return self.furniture_id
    
    def _get_furniture_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current position and orientation of furniture."""
        pos, orn = p.getBasePositionAndOrientation(self.furniture_id, physicsClientId=self.physics_client)
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
                 rotation_threshold: float = 0.1,
                 load_test: bool = False,
                 load_mass_kg: float = 2.27,  # 5 lbs
                 load_surface_idx: Optional[int] = None) -> SimulationResult:
        """
        Simulate furniture and test stability.
        
        Args:
            design: FurnitureDesign to test
            duration: Simulation duration in seconds
            stability_threshold: Max position change (m) to be considered stable
            rotation_threshold: Max rotation change (rad) to be considered stable
            load_test: If True, place a test load on the surface
            load_mass_kg: Mass of test load (default 5 lbs = 2.27 kg)
            load_surface_idx: Index of component to place load on (auto-detect if None)
        
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
        
        # Run initial stabilization (let furniture settle)
        settle_steps = int(1.0 * 240)  # 1 second to settle
        for step in range(settle_steps):
            p.stepSimulation(physicsClientId=self.physics_client)
            if self.gui:
                time.sleep(1/240)
        
        # Load test setup
        load_id = None
        load_initial_height = 0.0
        load_test_passed = True
        load_final_height = 0.0
        
        if load_test:
            # Find the flat surface to place load on
            surface_idx = load_surface_idx
            
            if surface_idx is None:
                # Auto-detect: find highest horizontal surface
                highest_z = -float('inf')
                for idx, comp in enumerate(design.components):
                    dims = comp.get_dimensions()
                    # Check if horizontal (thickness << width, height)
                    if dims[2] < dims[0] * 0.5 and dims[2] < dims[1] * 0.5:
                        top_z = comp.position[2] + dims[2]
                        if top_z > highest_z:
                            highest_z = top_z
                            surface_idx = idx
            
            if surface_idx is not None:
                # Get surface position
                comp = design.components[surface_idx]
                dims = comp.get_dimensions()
                
                # Place load on top of surface, centered
                load_x = (comp.position[0] + dims[0] / 2) / 1000.0
                load_y = (comp.position[1] + dims[1] / 2) / 1000.0
                load_z = (comp.position[2] + dims[2]) / 1000.0 + 0.05  # 5cm above surface
                
                # Create a simple box as the test load (10cm cube)
                load_size = 0.1  # 10cm
                load_collision = p.createCollisionShape(
                    p.GEOM_BOX, 
                    halfExtents=[load_size/2, load_size/2, load_size/2],
                    physicsClientId=self.physics_client
                )
                load_visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[load_size/2, load_size/2, load_size/2],
                    rgbaColor=[1, 0, 0, 1],  # Red cube
                    physicsClientId=self.physics_client
                )
                load_id = p.createMultiBody(
                    baseMass=load_mass_kg,
                    baseCollisionShapeIndex=load_collision,
                    baseVisualShapeIndex=load_visual,
                    basePosition=[load_x, load_y, load_z],
                    physicsClientId=self.physics_client
                )
                
                load_initial_height = load_z
        
        # Run main simulation
        steps = int(duration * 240)  # 240 Hz simulation
        for step in range(steps):
            p.stepSimulation(physicsClientId=self.physics_client)
            
            if self.gui:
                time.sleep(1/240)  # Real-time for visualization
            
            # Early exit if furniture falls below ground
            current_pos, _ = self._get_furniture_state()
            if current_pos[2] < -0.5:  # Fell through ground somehow
                break
        
        # Get final state
        final_pos, final_orn = self._get_furniture_state()
        
        # Check load test results
        if load_id is not None:
            load_pos, _ = p.getBasePositionAndOrientation(load_id, physicsClientId=self.physics_client)
            load_final_height = load_pos[2]
            
            # Load test fails if:
            # - Load fell to ground (height < 5cm)
            # - Load fell off the side (large XY displacement from furniture)
            load_fell_to_ground = load_final_height < 0.05
            
            furniture_center_x = (design.components[0].position[0] + 
                                  design.components[0].get_dimensions()[0] / 2) / 1000.0
            furniture_center_y = (design.components[0].position[1] + 
                                  design.components[0].get_dimensions()[1] / 2) / 1000.0
            load_displacement = np.sqrt(
                (load_pos[0] - furniture_center_x)**2 + 
                (load_pos[1] - furniture_center_y)**2
            )
            load_fell_off = load_displacement > 1.0  # More than 1m from center
            
            load_test_passed = not load_fell_to_ground and not load_fell_off
            
            # Clean up load
            p.removeBody(load_id, physicsClientId=self.physics_client)
        
        # Compute metrics
        position_change = np.linalg.norm(final_pos[:2] - initial_pos[:2])  # XY only
        rotation_change = self._quaternion_angle(initial_orn, final_orn)
        height_drop = initial_height - final_pos[2]
        
        # Determine if fell over (large rotation)
        fell_over = rotation_change > 0.5  # ~30 degrees
        
        # Check if something touched ground that shouldn't
        # (simplified - just check if height dropped significantly)
        touched_ground = height_drop > 0.1
        
        # Overall stability (includes load test if performed)
        stable = (
            position_change < stability_threshold and
            rotation_change < rotation_threshold and
            not fell_over and
            final_pos[2] > -0.01 and  # Allow slight ground penetration (1cm tolerance)
            load_test_passed  # Must pass load test if performed
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
            load_test_passed=load_test_passed,
            load_final_height=load_final_height,
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
    from furniture import Component, ComponentType, Joint, JointType, AssemblyGraph, FurnitureDesign

    def _make_test_table():
        """Simple 4-leg table for testing."""
        design = FurnitureDesign(name="test_table")
        # Tabletop
        design.add_component(Component(
            name="top", type=ComponentType.PANEL,
            profile=[(0, 0), (800, 0), (800, 500), (0, 500)],
            thickness=19.0, material="plywood",
            position=np.array([0.0, 0.0, 700.0]),
        ))
        # Four legs
        for i, (x, y) in enumerate([(20, 20), (720, 20), (20, 420), (720, 420)]):
            design.add_component(Component(
                name=f"leg_{i}", type=ComponentType.LEG,
                profile=[(0, 0), (60, 0), (60, 60), (0, 60)],
                thickness=700.0, material="plywood",
                position=np.array([float(x), float(y), 0.0]),
            ))
        return design

    print("Testing physics simulation...")
    print("=" * 50)

    table = _make_test_table()
    result = quick_stability_test(table, duration=2.0, gui=False)

    print(f"   Stable: {result.stable}")
    print(f"   Position change: {result.position_change:.4f} m")
    print(f"   Rotation change: {np.degrees(result.rotation_change):.2f} degrees")
    print(f"   Height drop: {result.height_drop:.4f} m")
    print(f"   Fell over: {result.fell_over}")
    print(f"   Wall time: {result.wall_time:.2f} s")

    print("\n" + "=" * 50)
    print("Simulation testing complete!")
