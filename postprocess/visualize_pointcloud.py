import numpy as np
import meshio
import struct
import os
import glob
import yaml


def read_binary_solution(bin_file_path):
    """Read solution vector from binary file."""
    with open(bin_file_path, 'rb') as f:
        size_bytes = f.read(4)
        size = struct.unpack('I', size_bytes)[0]
        data = np.frombuffer(f.read(size * 8), dtype=np.float64)
    return data


def read_dof_mapping(mapping_file):
    """Read DOF index to coordinate mapping.
    
    Returns:
        dict: mapping from DOF index to (x, y) coordinates
    """
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            dof_idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            mapping[dof_idx] = (x, y)
    return mapping


def main():
    """Main visualization function."""
    
    # Load configuration from YAML
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract paths from config
    dof_mapping_file = config['io']['dof_mapping_file']
    solution_dir = config['io']['solution_dir']
    output_dir = config['io']['output_dir']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read DOF mapping
    print(f"Reading DOF mapping from {dof_mapping_file}...")
    dof_mapping = read_dof_mapping(dof_mapping_file)
    print(f"Total DOFs: {len(dof_mapping)}")
    
    # Extract unique node coordinates (each node appears 3 times for u, v, p)
    n_dofs = len(dof_mapping)
    n_nodes = n_dofs // 3  # Each node has 3 DOFs (u, v, p)
    print(f"Number of nodes: {n_nodes}")
    
    # Build coordinate array from mapping
    # dof_mapping assumes DOF order is u0, v0, p0, u1, v1, p1, ...
    nodes = np.zeros((n_nodes, 3))  # (x, y, z) - z will be 0
    
    for node_idx in range(n_nodes):
        dof_idx = 3 * node_idx  
        if dof_idx in dof_mapping:
            x, y = dof_mapping[dof_idx]
            nodes[node_idx] = [x, y, 0.0]
    
    print(f"First 3 nodes:\n{nodes[:3]}")
    print(f"Last 3 nodes:\n{nodes[-3:]}")
    
    # Find all solution files
    solution_files = sorted(glob.glob(os.path.join(solution_dir, "solution_*.bin")))
    print(f"\nFound {len(solution_files)} solution files")
    
    if not solution_files:
        print(f"No solution files found in {solution_dir}")
        return
    
    # Process each solution file
    for sol_file in solution_files:
        try:
            # Extract step number from filename
            basename = os.path.basename(sol_file)
            step_num = basename.split('_')[1].split('.')[0]
            
            print(f"\nProcessing {basename}...")
            
            # Read solution vector
            sol_data = read_binary_solution(sol_file)
            print(f"  Solution data size: {len(sol_data)}")
            
            # Check data size: expect 3x nodes (u, v, p)
            if len(sol_data) != 3 * n_nodes:
                print(f"  ERROR: Solution size {len(sol_data)} incompatible with expected {3*n_nodes}")
                continue
            
            # Extract components (interleaved format: u, v, p, u, v, p, ...)
            u = sol_data[0::3]
            v = sol_data[1::3]
            p = sol_data[2::3]
            
            print(f"  u range: [{u.min():.6e}, {u.max():.6e}]")
            print(f"  v range: [{v.min():.6e}, {v.max():.6e}]")
            print(f"  p range: [{p.min():.6e}, {p.max():.6e}]")
            
            # Create velocity vector field (u, v, 0)
            vel = np.column_stack([u, v, np.zeros(n_nodes)])
            
            # Create point cloud mesh (vertices only)
            cells = [("vertex", np.arange(n_nodes).reshape(-1, 1))]
            
            # Add solution as point data
            point_data = {
                "velocity": vel,
                "pressure": p,
            }
            
            # Create mesh
            mesh = meshio.Mesh(
                points=nodes,
                cells=cells,
                point_data=point_data
            )
            
            # Write VTU file
            output_file = os.path.join(output_dir, f"solution_{step_num}.vtu")
            meshio.write(output_file, mesh, binary=True)
            print(f"  Written to {output_file}")
            
        except Exception as e:
            print(f"  Error processing {sol_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nAll VTU files written to {output_dir}")


if __name__ == "__main__":
    main()
