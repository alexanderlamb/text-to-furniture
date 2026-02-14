#!/usr/bin/env python3
"""
Interactive visualization for furniture simulations.

Usage:
    python scripts/visualize.py              # Random table
    python scripts/visualize.py --type shelf # Random shelf
    python scripts/visualize.py --interactive # Interactive mode with UI controls
"""
import sys
import time
import argparse
sys.path.insert(0, 'src')

import pybullet as p
from simulator import FurnitureSimulator
from genome import create_random_genome, create_table_genome, create_shelf_genome
from freeform_genome import create_freeform_genome
from urdf_generator import save_urdf_temp
import os


class InteractiveViewer:
    """
    Interactive furniture viewer with debug UI controls.
    """
    
    def __init__(self):
        self.sim = FurnitureSimulator(gui=True)
        self.sim._connect()
        self.furniture_id = None
        self.genome = None
        self.text_ids = []
        self.debug_lines = []
        self.is_stable = False  # Track stability state for hysteresis
        self.stable_frames = 0  # Count consecutive stable frames
        self.load_id = None  # 5lb test load
        self.design = None  # Current design for load placement
        
        # Set up camera
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-25,
            cameraTargetPosition=[0, 0, 0.4]
        )
        
        # Configure visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        
        # Add UI controls
        self._setup_ui()
    
    def _setup_ui(self):
        """Create debug UI sliders and buttons."""
        # Simulation controls
        self.gravity_slider = p.addUserDebugParameter("Gravity", -20, 0, -9.81)
        self.time_scale_slider = p.addUserDebugParameter("Time Scale", 0.1, 3.0, 1.0)
        
        # Camera controls  
        self.cam_distance_slider = p.addUserDebugParameter("Cam Distance", 0.5, 5.0, 2.5)
        self.cam_yaw_slider = p.addUserDebugParameter("Cam Yaw", -180, 180, 45)
        self.cam_pitch_slider = p.addUserDebugParameter("Cam Pitch", -90, 0, -25)
        
        # Action buttons (using sliders as triggers)
        self.new_table_btn = p.addUserDebugParameter("New Table", 1, 0, 0)
        self.new_shelf_btn = p.addUserDebugParameter("New Shelf", 1, 0, 0)
        self.new_freeform_btn = p.addUserDebugParameter("New Freeform", 1, 0, 0)
        self.reset_btn = p.addUserDebugParameter("Reset Position", 1, 0, 0)
        self.apply_force_btn = p.addUserDebugParameter("Push (apply force)", 1, 0, 0)
        self.drop_load_btn = p.addUserDebugParameter("Drop 5lb Load", 1, 0, 0)
        
        # Track button states
        self.last_table_val = 0
        self.last_shelf_val = 0
        self.last_freeform_val = 0
        self.last_reset_val = 0
        self.last_force_val = 0
        self.last_drop_load_val = 0
    
    def _clear_text(self):
        """Remove all debug text."""
        for text_id in self.text_ids:
            p.removeUserDebugItem(text_id)
        self.text_ids = []
    
    def _clear_lines(self):
        """Remove all debug lines."""
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
    
    def _add_text(self, text: str, position: list, color: list = [1, 1, 1], size: float = 1.5):
        """Add debug text at position."""
        text_id = p.addUserDebugText(
            text, position, 
            textColorRGB=color,
            textSize=size
        )
        self.text_ids.append(text_id)
        return text_id
    
    def _draw_axes(self, position: list = [0, 0, 0], length: float = 0.3):
        """Draw XYZ axes at position."""
        # X axis - red
        self.debug_lines.append(p.addUserDebugLine(
            position, [position[0] + length, position[1], position[2]],
            [1, 0, 0], lineWidth=2
        ))
        # Y axis - green  
        self.debug_lines.append(p.addUserDebugLine(
            position, [position[0], position[1] + length, position[2]],
            [0, 1, 0], lineWidth=2
        ))
        # Z axis - blue
        self.debug_lines.append(p.addUserDebugLine(
            position, [position[0], position[1], position[2] + length],
            [0, 0, 1], lineWidth=2
        ))
    
    def _draw_bounding_box(self):
        """Draw bounding box around furniture."""
        if self.furniture_id is None:
            return
        
        aabb_min, aabb_max = p.getAABB(self.furniture_id)
        
        # Draw 12 edges of bounding box
        corners = [
            [aabb_min[0], aabb_min[1], aabb_min[2]],
            [aabb_max[0], aabb_min[1], aabb_min[2]],
            [aabb_max[0], aabb_max[1], aabb_min[2]],
            [aabb_min[0], aabb_max[1], aabb_min[2]],
            [aabb_min[0], aabb_min[1], aabb_max[2]],
            [aabb_max[0], aabb_min[1], aabb_max[2]],
            [aabb_max[0], aabb_max[1], aabb_max[2]],
            [aabb_min[0], aabb_max[1], aabb_max[2]],
        ]
        edges = [
            (0,1), (1,2), (2,3), (3,0),  # bottom
            (4,5), (5,6), (6,7), (7,4),  # top
            (0,4), (1,5), (2,6), (3,7),  # verticals
        ]
        for i, j in edges:
            self.debug_lines.append(p.addUserDebugLine(
                corners[i], corners[j], [1, 1, 0], lineWidth=1
            ))
    
    def _remove_load(self):
        """Remove the test load if present."""
        if self.load_id is not None:
            try:
                p.removeBody(self.load_id)
            except:
                pass
            self.load_id = None
    
    def _drop_load(self):
        """Drop a 5lb test load on the highest flat surface."""
        if self.design is None or self.furniture_id is None:
            return
        
        # Remove existing load
        self._remove_load()
        
        # Find highest horizontal surface
        highest_z = -float('inf')
        surface_comp = None
        
        for comp in self.design.components:
            dims = comp.get_dimensions()
            # Check if horizontal (thickness << width, height) and large enough
            if dims[2] < dims[0] * 0.5 and dims[2] < dims[1] * 0.5:
                if dims[0] > 100 and dims[1] > 100:  # At least 10cm x 10cm
                    top_z = comp.position[2] + dims[2]
                    if top_z > highest_z:
                        highest_z = top_z
                        surface_comp = comp
        
        if surface_comp is None:
            print("No suitable flat surface found!")
            return
        
        # Get furniture current position offset
        furn_pos, _ = p.getBasePositionAndOrientation(self.furniture_id)
        
        # Calculate load position (center of surface)
        dims = surface_comp.get_dimensions()
        load_x = furn_pos[0] + (surface_comp.position[0] + dims[0] / 2) / 1000.0
        load_y = furn_pos[1] + (surface_comp.position[1] + dims[1] / 2) / 1000.0
        load_z = furn_pos[2] + (surface_comp.position[2] + dims[2]) / 1000.0 + 0.15  # 15cm above
        
        # Create 5lb (2.27kg) test load - red 10cm cube
        load_mass = 2.27  # 5 lbs in kg
        load_size = 0.1  # 10cm cube
        
        load_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[load_size/2, load_size/2, load_size/2]
        )
        load_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[load_size/2, load_size/2, load_size/2],
            rgbaColor=[1, 0, 0, 0.8]  # Red, slightly transparent
        )
        
        self.load_id = p.createMultiBody(
            baseMass=load_mass,
            baseCollisionShapeIndex=load_collision,
            baseVisualShapeIndex=load_visual,
            basePosition=[load_x, load_y, load_z]
        )
        
        print(f"Dropped 5lb load at ({load_x:.2f}, {load_y:.2f}, {load_z:.2f})")
    
    def load_genome(self, genome, name: str = "Furniture"):
        """Load a genome into the simulation."""
        # Remove existing furniture and load
        if self.furniture_id is not None:
            p.removeBody(self.furniture_id)
        self._remove_load()
        
        self._clear_text()
        self._clear_lines()
        
        # Reset stability tracking
        self.is_stable = False
        self.stable_frames = 0
        
        self.genome = genome
        design = genome.to_furniture_design()
        self.design = design  # Store for load placement
        
        # Calculate spawn height based on furniture's lowest point
        # We want the bottom of the furniture to start just above ground
        min_z = float('inf')
        for comp in design.components:
            comp_bottom_z = comp.position[2]  # Component's z position (in mm)
            min_z = min(min_z, comp_bottom_z)
        
        # Convert to meters and add small offset above ground
        # If min_z is 0, spawn at 0.02m so it drops slightly
        # If min_z is negative (shouldn't happen), compensate
        spawn_z = 0.02 - (min_z / 1000.0) if min_z != float('inf') else 0.5
        spawn_z = max(spawn_z, 0.02)  # Minimum spawn height
        
        # Load URDF
        urdf_path = save_urdf_temp(design)
        self.furniture_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, spawn_z],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,  # Prevent internal collisions
        )
        os.unlink(urdf_path)
        
        # Add labels
        self._add_text(name, [0, 0, 1.5], [0, 1, 0], 2.0)
        
        # Count components
        n_components = len([c for c in genome.components if c.active])
        self._add_text(f"Components: {n_components}", [0, 0, 1.3], [0.8, 0.8, 0.8], 1.2)
        
        # Draw axes at origin
        self._draw_axes()
        
        print(f"\nLoaded: {name}")
        print(f"  Components: {n_components}")
    
    def _update_camera(self):
        """Update camera from slider values."""
        distance = p.readUserDebugParameter(self.cam_distance_slider)
        yaw = p.readUserDebugParameter(self.cam_yaw_slider)
        pitch = p.readUserDebugParameter(self.cam_pitch_slider)
        
        # Get current camera target (furniture position or origin)
        if self.furniture_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.furniture_id)
            target = [pos[0], pos[1], pos[2]]
        else:
            target = [0, 0, 0.4]
        
        p.resetDebugVisualizerCamera(distance, yaw, pitch, target)
    
    def _check_buttons(self):
        """Check if any button was clicked."""
        # New table button
        table_val = p.readUserDebugParameter(self.new_table_btn)
        if table_val > self.last_table_val:
            self.load_genome(create_table_genome(), "Random Table")
            self._draw_bounding_box()
        self.last_table_val = table_val
        
        # New shelf button
        shelf_val = p.readUserDebugParameter(self.new_shelf_btn)
        if shelf_val > self.last_shelf_val:
            self.load_genome(create_shelf_genome(), "Random Shelf")
            self._draw_bounding_box()
        self.last_shelf_val = shelf_val
        
        # New freeform button
        freeform_val = p.readUserDebugParameter(self.new_freeform_btn)
        if freeform_val > self.last_freeform_val:
            self.load_genome(create_freeform_genome(), "Freeform")
            self._draw_bounding_box()
        self.last_freeform_val = freeform_val
        
        # Reset button
        reset_val = p.readUserDebugParameter(self.reset_btn)
        if reset_val > self.last_reset_val and self.furniture_id is not None:
            p.resetBasePositionAndOrientation(
                self.furniture_id, [0, 0, 0.5], [0, 0, 0, 1]
            )
            p.resetBaseVelocity(self.furniture_id, [0, 0, 0], [0, 0, 0])
        self.last_reset_val = reset_val
        
        # Apply force button
        force_val = p.readUserDebugParameter(self.apply_force_btn)
        if force_val > self.last_force_val and self.furniture_id is not None:
            # Apply random horizontal force
            import random
            force = [random.uniform(-100, 100), random.uniform(-100, 100), 0]
            pos, _ = p.getBasePositionAndOrientation(self.furniture_id)
            p.applyExternalForce(
                self.furniture_id, -1, force, pos, p.WORLD_FRAME
            )
            print(f"Applied force: [{force[0]:.1f}, {force[1]:.1f}, 0]")
        self.last_force_val = force_val
        
        # Drop load button
        drop_load_val = p.readUserDebugParameter(self.drop_load_btn)
        if drop_load_val > self.last_drop_load_val:
            self._drop_load()
        self.last_drop_load_val = drop_load_val
    
    def _update_status(self):
        """Update status display."""
        if self.furniture_id is None:
            return
        
        # Get state
        pos, orn = p.getBasePositionAndOrientation(self.furniture_id)
        vel, ang_vel = p.getBaseVelocity(self.furniture_id)
        
        # Check velocity
        speed = (vel[0]**2 + vel[1]**2 + vel[2]**2) ** 0.5
        
        # Hysteresis to prevent flickering:
        # - Need 30 consecutive slow frames to become STABLE
        # - Need speed > 0.05 to become MOVING again
        if speed < 0.01 and pos[2] > 0:
            self.stable_frames += 1
            if self.stable_frames > 30:  # ~0.125 seconds at 240Hz
                self.is_stable = True
        else:
            self.stable_frames = 0
            if speed > 0.05:  # Higher threshold to leave stable state
                self.is_stable = False
        
        # Update status text (remove old, add new)
        self._clear_text()
        
        status = "STABLE" if self.is_stable else "MOVING"
        color = [0, 1, 0] if self.is_stable else [1, 0.5, 0]
        self._add_text(status, [pos[0], pos[1], pos[2] + 0.8], color, 1.5)
        
        # Height indicator
        self._add_text(f"h={pos[2]:.2f}m", [pos[0] + 0.3, pos[1], pos[2] + 0.6], [0.7, 0.7, 0.7], 1.0)
    
    def run(self):
        """Main loop."""
        print("\n" + "="*50)
        print("Interactive Furniture Viewer")
        print("="*50)
        print("\nUse sliders in the right panel to control:")
        print("  - Gravity strength")
        print("  - Time scale (simulation speed)")
        print("  - Camera position")
        print("\nClick buttons (sliders) to:")
        print("  - Generate new table/shelf")
        print("  - Reset position")
        print("  - Apply random force (stability test)")
        print("\nMouse controls:")
        print("  - Drag to rotate view")
        print("  - Scroll to zoom")
        print("  - Ctrl+drag to pan")
        print("\nPress Ctrl+C in terminal to exit.")
        print("="*50)
        
        # Load initial table
        self.load_genome(create_table_genome(), "Random Table")
        self._draw_bounding_box()
        
        try:
            while True:
                # Check if window was closed
                if not p.isConnected():
                    print("\nWindow closed.")
                    break
                
                # Read gravity from slider
                gravity = p.readUserDebugParameter(self.gravity_slider)
                p.setGravity(0, 0, gravity)
                
                # Read time scale
                time_scale = p.readUserDebugParameter(self.time_scale_slider)
                
                # Check button presses
                self._check_buttons()
                
                # Update camera
                self._update_camera()
                
                # Step simulation
                p.stepSimulation()
                
                # Update status every few frames
                self._update_status()
                
                # Sleep based on time scale
                time.sleep(1.0 / (240 * time_scale))
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            # Catch errors from window being closed mid-operation
            if "Not connected" in str(e) or "physics server" in str(e).lower():
                print("\nWindow closed.")
            else:
                raise
        finally:
            if p.isConnected():
                p.disconnect()


def visualize_design(sim: FurnitureSimulator, genome, name: str, duration: float = 5.0):
    """Visualize a single design (batch mode)."""
    design = genome.to_furniture_design()
    print(f"\n{'='*50}")
    print(f"Simulating: {name}")
    print(f"Components: {len([c for c in genome.components if c.active])}")
    print(f"{'='*50}")
    
    result = sim.simulate(design, duration=duration)
    
    print(f"\nResult:")
    print(f"  Stable: {result.stable}")
    print(f"  Position change: {result.position_change:.4f}m")
    print(f"  Rotation change: {result.rotation_change:.4f} rad")
    print(f"  Fell over: {result.fell_over}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Visualize furniture physics simulation')
    parser.add_argument('--type', choices=['table', 'shelf', 'mixed'], default='table',
                        help='Type of furniture to generate (batch mode)')
    parser.add_argument('--batch', type=int, default=None,
                        help='Number of designs to simulate in batch mode')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Simulation duration in seconds (batch mode)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run interactive mode with UI controls')
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive or args.batch is None:
        viewer = InteractiveViewer()
        viewer.run()
        return
    
    # Batch mode (original behavior)
    print("Starting PyBullet GUI...")
    sim = FurnitureSimulator(gui=True)
    
    stable_count = 0
    
    for i in range(args.batch):
        # Generate design based on type
        if args.type == 'table':
            genome = create_table_genome()
            name = f"Table #{i+1}"
        elif args.type == 'shelf':
            genome = create_shelf_genome()
            name = f"Shelf #{i+1}"
        else:  # mixed
            import random
            if random.random() < 0.5:
                genome = create_table_genome()
                name = f"Table #{i+1}"
            else:
                genome = create_shelf_genome()
                name = f"Shelf #{i+1}"
        
        result = visualize_design(sim, genome, name, args.duration)
        if result.stable:
            stable_count += 1
        
        if i < args.batch - 1:
            input("\nPress Enter for next design...")
    
    print(f"\n{'='*50}")
    print(f"Final: {stable_count}/{args.batch} stable ({100*stable_count/args.batch:.0f}%)")
    print(f"{'='*50}")
    
    input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
