import open3d as o3d
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="View a single PLY file.")
    parser.add_argument("path", help="Path to the PLY file")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: File not found at {args.path}")
        sys.exit(1)

    print(f"Loading {args.path}...")
    try:
        mesh = o3d.io.read_triangle_mesh(args.path)
        if not mesh.has_vertices():
            print("Warning: The mesh has no vertices. It might be empty or not a valid triangle mesh.")
        
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        print("Opening viewer... (Press 'Q' to quit, 'H' for help)")
        o3d.visualization.draw_geometries([mesh], window_name=f"PLY Viewer - {os.path.basename(args.path)}", width=1024, height=768)

    except Exception as e:
        print(f"Error loading or visualizing PLY: {e}")

if __name__ == "__main__":
    main()
