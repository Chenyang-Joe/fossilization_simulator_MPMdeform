
import os
import numpy as np
import trimesh
import argparse
import time
from pathlib import Path

def jiggle_meshes(input_folder, output_folder, global_jiggle_range=0.05, max_rotation_degrees=5.0):
    """
    Jiggle mesh vertices based on relative bounding box sizes.
    
    Args:
        input_folder: Path to folder containing input meshes
        output_folder: Path to folder for saving jiggled meshes
        global_jiggle_range: Global jiggling scale factor
        max_rotation_degrees: Maximum rotation angle in degrees
    """
    # Set random seed based on current time
    np.random.seed(int(time.time()))
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Read all meshes and get supported file extensions
    mesh_extensions = ['.obj', '.ply', '.stl', '.off']
    mesh_files = []
    for ext in mesh_extensions:
        mesh_files.extend(Path(input_folder).glob(f'*{ext}'))
        mesh_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    if not mesh_files:
        print(f"No mesh files found in {input_folder}")
        return
    
    print(f"Found {len(mesh_files)} mesh files")
    
    # Load all meshes
    meshes = []
    mesh_names = []
    for mesh_file in mesh_files:
        try:
            mesh = trimesh.load(str(mesh_file))
            if hasattr(mesh, 'vertices'):
                meshes.append(mesh)
                mesh_names.append(mesh_file.stem)
                print(f"Loaded: {mesh_file.name}")
            else:
                print(f"Warning: {mesh_file.name} doesn't contain vertices")
        except Exception as e:
            print(f"Error loading {mesh_file.name}: {e}")
    
    if not meshes:
        print("No valid meshes loaded")
        return
    
    # 2. Calculate overall bounding box by concatenating all vertices
    all_vertices = []
    for mesh in meshes:
        all_vertices.append(mesh.vertices)
    
    combined_vertices = np.vstack(all_vertices)
    overall_bbox_min = np.min(combined_vertices, axis=0)
    overall_bbox_max = np.max(combined_vertices, axis=0)
    overall_bbox_size = np.linalg.norm(overall_bbox_max - overall_bbox_min)
    
    print(f"Overall bounding box size: {overall_bbox_size:.4f}")
    
    # 3. Jiggle each mesh and save
    for i, (mesh, mesh_name) in enumerate(zip(meshes, mesh_names)):
        # Calculate individual mesh bounding box
        mesh_bbox_min = np.min(mesh.vertices, axis=0)
        mesh_bbox_max = np.max(mesh.vertices, axis=0)
        mesh_bbox_size = np.linalg.norm(mesh_bbox_max - mesh_bbox_min)
        
        # Calculate jiggle range based on relative size
        relative_size = mesh_bbox_size / overall_bbox_size
        jiggle_range = relative_size * global_jiggle_range
        
        print(f"Mesh {mesh_name}: bbox_size={mesh_bbox_size:.4f}, relative_size={relative_size:.4f}, jiggle_range={jiggle_range:.6f}")
        
        # Apply jiggling - same displacement and rotation for all vertices in this fragment
        jiggled_vertices = mesh.vertices.copy()
        
        # Calculate fragment center from bounding box
        fragment_center = (mesh_bbox_min + mesh_bbox_max) / 2
        
        # 1. Apply random rotation around fragment center
        # Generate random rotation axis uniformly distributed on unit sphere
        # Method: sample from normal distribution and normalize
        rotation_axis = np.random.normal(0, 1, 3)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Generate random rotation angle between 0 and max_rotation_degrees
        rotation_angle = np.random.random() * max_rotation_degrees * np.pi / 180.0  # Convert to radians
        
        # Create rotation matrix using Rodrigues' rotation formula
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        cross_product_matrix = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = (cos_angle * np.eye(3) + 
                          sin_angle * cross_product_matrix + 
                          (1 - cos_angle) * np.outer(rotation_axis, rotation_axis))
        
        # Translate vertices to origin (relative to fragment center)
        centered_vertices = jiggled_vertices - fragment_center
        
        # Apply rotation
        rotated_vertices = centered_vertices @ rotation_matrix.T
        
        # Translate back to original position
        jiggled_vertices = rotated_vertices + fragment_center
        
        # 2. Apply random translation
        # Generate single random displacement for the entire fragment in range [-jiggle_range, jiggle_range]
        fragment_displacement = (np.random.random(3) - 0.5) * 2 * jiggle_range
        jiggled_vertices += fragment_displacement
        
        # Create new mesh with jiggled vertices
        jiggled_mesh = mesh.copy()
        jiggled_mesh.vertices = jiggled_vertices
        
        # Save jiggled mesh
        output_filename = f"{mesh_name}_jiggled{mesh_files[i].suffix}"
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            jiggled_mesh.export(output_path)
            print(f"Saved: {output_filename}")
        except Exception as e:
            print(f"Error saving {output_filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Jiggle mesh vertices based on relative bounding box sizes")
    parser.add_argument('--input_folder', type=str, help='Path to folder containing input meshes')
    parser.add_argument('--output_folder', type=str, help='Path to folder for saving jiggled meshes')
    parser.add_argument('--global_jiggle_range', type=float, default=0.001, 
                       help='Global jiggling scale factor (default: 0.001)')
    parser.add_argument('--max_rotation_degrees', type=float, default=5.0,
                       help='Maximum rotation angle in degrees (default: 5.0)')
    
    args = parser.parse_args()
    
    jiggle_meshes(args.input_folder, args.output_folder, args.global_jiggle_range, args.max_rotation_degrees)

if __name__ == "__main__":
    main()