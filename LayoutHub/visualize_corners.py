import os
import json
import argparse
import glob
import cv2
import numpy as np
from utils.panorama import draw_boundary_from_cor_id, lineIdxFromCors

def parse_layout(layout_path, img_w, img_h):
    """
    Parse layout file and return corner coordinates in pixel space.
    """
    ext = os.path.splitext(layout_path)[1].lower()
    
    if ext == '.json':
        with open(layout_path, 'r') as f:
            data = json.load(f)
        
        if 'uv' in data:
            # HorizonNet / DMH-Net format
            uv_norm = np.array(data['uv']) # Shape (N, 2), normalized 0-1
            
            # Convert normalized UV to pixel coordinates
            cor_id = np.zeros_like(uv_norm)
            cor_id[:, 0] = uv_norm[:, 0] * img_w
            cor_id[:, 1] = uv_norm[:, 1] * img_h
            return cor_id
            
        elif 'layoutPoints' in data:
            # LGT-Net format
            # Extract parameters
            # LGT-Net y is vertical down? (y=1.6 floor).
            # We assume camera at (0,0,0).
            # Floor y = 1.6. Ceiling y = -cameraCeilingHeight.
            
            camera_height = data.get('cameraHeight', 1.6)
            camera_ceiling_height = data.get('cameraCeilingHeight', 1.6) # Default to symmetric if missing
            
            points = data['layoutPoints']['points']
            # Sort points by ID just in case
            points.sort(key=lambda p: p['id'])
            
            # Check for VP file to handle rotation
            # JSON: {name}_pred.json -> VP: {name}_vp.txt
            # Fallback: {name}.json -> VP: {name}_vp.txt
            vp_path = layout_path.replace("_pred.json", "_vp.txt")
            if vp_path == layout_path:
                vp_path = layout_path.replace(".json", "_vp.txt")
            
            vp = None
            if os.path.exists(vp_path):
                try:
                    with open(vp_path, 'r') as f:
                        lines = f.readlines()
                        vp_data = [[float(x) for x in line.strip().split()] for line in lines]
                        vp = np.array(vp_data) # 3x3 matrix
                        vp = vp[::-1] # Reverse rows to match inference.py rotation logic
                    # print(f"Loaded VP from {vp_path}")
                except Exception as e:
                    print(f"Failed to load VP from {vp_path}: {e}")
            
            cor_list = []
            
            for p in points:
                # LGT-Net JSON coords: x, y, z. 
                # writer.py applies: x'=x, y'=y, z'=-z (relative to internal coords)
                # So Internal x = x_json, Internal y = y_json, Internal z = -z_json
                
                lgt_x_json, lgt_y_json, lgt_z_json = p['xyz']
                
                # 1. Convert to Internal Coords
                lgt_x = lgt_x_json
                lgt_y = lgt_y_json
                lgt_z = -lgt_z_json
                
                # 2. Apply Inverse Rotation if VP exists
                # Internal coords are in the rotated frame.
                # Original coords = Internal coords @ VP
                # (derived from xyzOld = xyzNew @ vp in rotatePanorama logic)
                
                vec = np.array([lgt_x, lgt_y, lgt_z])
                if vp is not None:
                    vec = vec @ vp
                    lgt_x, lgt_y, lgt_z = vec
                
                # 3. Map to Panorama Coords (HorizonNet convention)
                # LGT Internal: x=Right, y=Down, z=Forward
                # HorizonNet: x=Right, y=Forward, z=Up
                # So:
                # Pano X = LGT X
                # Pano Y = LGT Z
                # Pano Z = -LGT Y
                
                pano_x_floor = lgt_x
                pano_y_floor = lgt_z
                pano_z_floor = -lgt_y 
                
                # Ceiling Point
                # For ceiling, we need to be careful. 
                # The JSON points are usually floor corners (y ~ 1.6).
                # We need to project to ceiling.
                # In LGT-Net internal, ceiling is at y = -cameraCeilingHeight.
                # But we just rotated the *floor* point.
                # If the walls are vertical in the *aligned* frame, they might NOT be vertical in the *original* frame if the camera was tilted!
                # However, HorizonNet/LayoutHub assumes vertical walls (gravity aligned).
                # If LGT-Net aligned the image to gravity (using VP), then the "Internal" frame is gravity aligned.
                # So in the Internal frame, the ceiling point is simply (x, -ceil_h, z).
                # Then we rotate THAT point back to original frame.
                
                # Re-calculate Internal Ceiling Point
                # Floor point was (lgt_x_json, lgt_y_json, -lgt_z_json) [before rotation]
                # Actually lgt_y_json is roughly camera_height (1.6).
                # Ceiling y should be -cameraCeilingHeight.
                # So we take the floor point in Internal frame, change y, then rotate.
                
                lgt_x_internal = lgt_x_json
                lgt_z_internal = -lgt_z_json
                # lgt_y_internal_floor = lgt_y_json (approx 1.6)
                lgt_y_internal_ceil = -camera_ceiling_height
                
                vec_ceil = np.array([lgt_x_internal, lgt_y_internal_ceil, lgt_z_internal])
                if vp is not None:
                    vec_ceil = vec_ceil @ vp
                
                lgt_x_c, lgt_y_c, lgt_z_c = vec_ceil
                
                pano_x_ceil = lgt_x_c
                pano_y_ceil = lgt_z_c
                pano_z_ceil = -lgt_y_c
                
                # Convert to UV
                def xyz2uv_single(x, y, z):
                    normXY = np.sqrt(x**2 + y**2)
                    normXYZ = np.sqrt(x**2 + y**2 + z**2)
                    v = np.arcsin(z / normXYZ) if normXYZ > 0 else 0
                    u = np.arctan2(x, y) # panorama.py uses arctan2(x, y) for u
                    return u, v

                u_floor, v_floor = xyz2uv_single(pano_x_floor, pano_y_floor, pano_z_floor)
                u_ceil, v_ceil = xyz2uv_single(pano_x_ceil, pano_y_ceil, pano_z_ceil)
                
                # Convert UV to Image Coords
                def uv2img(u, v, w, h):
                    x = (u / (2 * np.pi) + 0.5) * w
                    y = h * (0.5 - v / np.pi)
                    return x, y

                img_x_floor, img_y_floor = uv2img(u_floor, v_floor, img_w, img_h)
                img_x_ceil, img_y_ceil = uv2img(u_ceil, v_ceil, img_w, img_h)
                
                # Append [Ceiling, Floor] pair
                cor_list.append([img_x_ceil, img_y_ceil])
                cor_list.append([img_x_floor, img_y_floor])
                
            return np.array(cor_list)
            
        else:
            print(f"Skipping {layout_path}: Unknown JSON format")
            return None
        
    elif ext == '.txt':
        try:
            # Assuming TXT format is "x y" per line in pixel coordinates
            # HoHoNet outputs in 1024x512 resolution
            cor_id = np.loadtxt(layout_path)
            
            # Scale to current image size
            # HoHoNet default output size
            HOHO_W, HOHO_H = 1024, 512
            
            cor_id[:, 0] = cor_id[:, 0] / HOHO_W * img_w
            cor_id[:, 1] = cor_id[:, 1] / HOHO_H * img_h
            
            return cor_id
        except Exception as e:
            print(f"Error parsing TXT {layout_path}: {e}")
            return None
    else:
        print(f"Unsupported layout format: {ext}")
        return None

def draw_boundary_thicker(cor_id, img_src, thickness=3):
    """
    Draw layout boundary with thicker lines and blue color.
    """
    im_h, im_w = img_src.shape[:2]
    cor_all = [cor_id]
    for i in range(len(cor_id)):
        cor_all.append(cor_id[i, :])
        cor_all.append(cor_id[(i+2) % len(cor_id), :])
    cor_all = np.vstack(cor_all)

    rs, cs = lineIdxFromCors(cor_all, im_w, im_h)
    rs = np.array(rs)
    cs = np.array(cs)

    panoEdgeC = img_src.copy()
    
    # Blue color in RGB is (0, 0, 255)
    # We want to draw thicker lines.
    # We can dilate the mask of lines.
    
    mask = np.zeros((im_h, im_w), dtype=np.uint8)
    mask[np.clip(rs, 0, im_h - 1), np.clip(cs, 0, im_w - 1)] = 255
    
    # Dilate to make it thicker
    kernel_size = thickness
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    
    # Apply blue color where mask is present
    # RGB: Blue=(0, 0, 255)
    panoEdgeC[mask_dilated > 0] = [0, 0, 255]

    return panoEdgeC

def visualize_layout(layout_path, image_path, output_path):
    # Read Image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
        
    img_src = cv2.imread(image_path)
    if img_src is None:
        print(f"Failed to read image: {image_path}")
        return
        
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    H, W = img_src.shape[:2]
    
    # Parse Layout
    cor_id = parse_layout(layout_path, W, H)
    if cor_id is None:
        return

    # Draw
    try:
        # img_viz = draw_boundary_from_cor_id(cor_id, img_src)
        img_viz = draw_boundary_thicker(cor_id, img_src, thickness=5)
        
        # Save
        # Convert back to BGR for saving
        img_viz_bgr = cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_viz_bgr)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error processing {layout_path}: {e}")

def process_single_pair(layout_path, image_path, output_path):
    if not layout_path or not image_path or not output_path:
        print("Error: --layout, --image, and --out are required for single file mode.")
        return
    visualize_layout(layout_path, image_path, output_path)

def process_directory(layout_dir, image_dir, output_dir):
    if not layout_dir or not output_dir:
        print("Error: --layout_dir and --output_dir are required for directory mode.")
        return

    if image_dir is None:
        image_dir = layout_dir

    os.makedirs(output_dir, exist_ok=True)

    # Find all JSON and TXT files
    layout_files = glob.glob(os.path.join(layout_dir, "*.json")) + glob.glob(os.path.join(layout_dir, "*.txt"))
    
    if not layout_files:
        print(f"No layout files (json/txt) found in {layout_dir}")
        return

    print(f"Found {len(layout_files)} layout files.")

    for layout_file in layout_files:
        basename = os.path.basename(layout_file)
        # Try to find corresponding image
        # Support both .json and .txt extensions for layout files
        file_root = os.path.splitext(basename)[0]
        
        # Try likely image candidates
        candidates = [
            file_root + ".jpg",
            file_root + ".png",
            file_root + ".raw.png", # Legacy support
            basename.replace(".json", ".raw.png") # Legacy specific
        ]
        
        img_path = None
        for cand in candidates:
            cand_path = os.path.join(image_dir, cand)
            if os.path.exists(cand_path):
                img_path = cand_path
                break
        
        if img_path is None:
            # Fallback to constructing a default path (even if it doesn't exist, visualize_layout handles check)
            # Default to .jpg since that seems to be the GT case
            img_path = os.path.join(image_dir, file_root + ".jpg")
        
        output_filename = file_root + "_viz.png"
        output_path = os.path.join(output_dir, output_filename)
        
        visualize_layout(layout_file, img_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Visualize Layout from JSON or TXT corners")
    
    # Single file mode arguments
    parser.add_argument("--layout", help="Path to a single layout file (JSON or TXT)")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--out", help="Path to output image file")
    
    # Directory mode arguments
    parser.add_argument("--layout_dir", help="Directory containing layout JSON files")
    parser.add_argument("--image_dir", help="Directory containing images (defaults to layout_dir)")
    parser.add_argument("--output_dir", help="Directory to save results")
    
    args = parser.parse_args()

    if args.layout:
        process_single_pair(args.layout, args.image, args.out)
    elif args.layout_dir:
        process_directory(args.layout_dir, args.image_dir, args.output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
