import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# from HorizonNet.misc... -> from utils...
try:
    from utils.post_proc import np_coor2xy
    from utils.panostretch import pano_connect_points
except ImportError:
    print("Error: Could not import utils. Make sure you run this from the LayoutHub directory.")
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HorizonNet layout: Panorama + Top-Down Floor Plan")
    parser.add_argument('--layout', required=True, help='Path to HorizonNet output JSON')
    parser.add_argument('--image', required=True, help='Path to original panoramic image')
    parser.add_argument('--out', required=True, help='Path to output visualization file')
    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_color(idx, total, mode='hsv'):
    """Generate a color for index i out of total."""
    # 使用 hsv 色彩映射
    x = idx / max(1, total)
    # RGBA
    c = cm.hsv(x)
    return c

def visualize_combined(args):
    # 1. Load Data
    data = load_json(args.layout)
    img_src = cv2.imread(args.image)
    if img_src is None:
        print(f"Error: Could not read image {args.image}")
        return
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    
    H, W = img_src.shape[:2]
    
    # 2. Extract UV and decide format
    if 'uv' not in data:
        print("Error: JSON must contain 'uv' key.")
        return
        
    uv = np.array(data['uv']) # N x 2, normalized 0-1
    z1 = data.get('z1', 50) # Floor depth, default 50
    
    # Determine if raw or post-processed
    is_raw = len(uv) > 100
    
    # Scale UV to image size
    uv_img = uv.copy()
    uv_img[:, 0] *= W
    uv_img[:, 1] *= H
    
    # 3. Process Data for Visualization
    
    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- Panorama View (ax1) ---
    ax1.imshow(img_src)
    ax1.set_title("Panorama View")
    ax1.axis('off')
    
    # --- Floor Plan View (ax2) ---
    ax2.set_title("Top-Down Floor Plan")
    ax2.axis('equal')
    ax2.grid(True)
    ax2.invert_yaxis()
    
    if not is_raw:
        # --- Post-processed (Corners) ---
        num_points = len(uv)
        num_corners = num_points // 2
        
        # Floor UV (odd indices)
        floor_indices = range(1, len(uv), 2)
        
        floor_uv_img = uv_img[floor_indices]
        
        # Project to XY
        # np_coor2xy defaults center at (floorW/2, floorH/2)
        # We set floorW, floorH to W, H, then offset to center at (0,0)
        
        floor_xy = np_coor2xy(floor_uv_img, z=z1, coorW=W, coorH=H, floorW=W, floorH=W) 
        
        # Correct center point: shift coordinates so camera is at (0,0)
        # np_coor2xy formula: x = c * sin(u) + floorW/2 - 0.5
        # So subtract floorW/2 - 0.5
        center_offset_x = W / 2 - 0.5
        center_offset_y = W / 2 - 0.5 # using floorH=W (square floor plan space)
        
        floor_xy[:, 0] -= center_offset_x
        floor_xy[:, 1] -= center_offset_y
        
        # Draw camera position (now at origin)
        ax2.scatter([0], [0], marker='^', color='k', s=100, label='Camera')
        
        # Iterate through segments
        for i in range(num_corners):
            c_idx = 2 * i
            f_idx = 2 * i + 1
            
            next_i = (i + 1) % num_corners
            next_c_idx = 2 * next_i
            next_f_idx = 2 * next_i + 1
            
            color = get_color(i, num_corners)
            
            # --- Draw on Panorama (use pano_connect_points for curves) ---
            
            # Ceiling Line i -> i+1
            p1_c = uv_img[c_idx] # [u, v] in pixel
            p2_c = uv_img[next_c_idx]
            
            # pano_connect_points needs pixel coordinates (x, y)
            # z parameter: Ceiling usually z=-50, Floor z=50
            # Assume -50 and 50 are generic values, but symbols matter.
            
            # Draw Ceiling Curve
            pts_c = pano_connect_points(p1_c, p2_c, z=-50, w=W, h=H)
            
            # Handle Wrap around
            pts_c_x = pts_c[:, 0] % W
            pts_c_y = pts_c[:, 1]
            
            # Find split points (Wrap point)
            diffs = np.abs(np.diff(pts_c_x))
            split_indices = np.where(diffs > W / 2)[0]
            
            if len(split_indices) > 0:
                splits = np.split(np.stack([pts_c_x, pts_c_y], axis=1), split_indices + 1)
                for seg in splits:
                     ax1.plot(seg[:, 0], seg[:, 1], color=color, linewidth=3)
            else:
                ax1.plot(pts_c_x, pts_c_y, color=color, linewidth=3)
                
            # Draw Floor Curve
            p1_f = uv_img[f_idx]
            p2_f = uv_img[next_f_idx]
            pts_f = pano_connect_points(p1_f, p2_f, z=50, w=W, h=H) # z=50 for floor
            
            pts_f_x = pts_f[:, 0] % W
            pts_f_y = pts_f[:, 1]
            
            diffs_f = np.abs(np.diff(pts_f_x))
            split_indices_f = np.where(diffs_f > W / 2)[0]
            
            if len(split_indices_f) > 0:
                splits = np.split(np.stack([pts_f_x, pts_f_y], axis=1), split_indices_f + 1)
                for seg in splits:
                     ax1.plot(seg[:, 0], seg[:, 1], color=color, linewidth=3)
            else:
                ax1.plot(pts_f_x, pts_f_y, color=color, linewidth=3)
            
            # Draw Vertical Line at Corner i
            x1_c, y1_c = p1_c
            x1_f, y1_f = p1_f
            ax1.plot([x1_c, x1_f], [y1_c, y1_f], color=color, linewidth=2, linestyle=':')
            
            # --- Draw on Floor Plan ---
            fx1, fy1 = floor_xy[i]
            fx2, fy2 = floor_xy[next_i]
            
            ax2.plot([fx1, fx2], [fy1, fy2], color=color, linewidth=3)
            ax2.scatter([fx1], [fy1], color=color, s=50, marker='o', edgecolors='black')
            
            # (Labels removed)

    else:
        # --- Raw Data (Dense) ---
        floor_uv_indices = range(1, len(uv), 2)
        floor_uv_img = uv_img[floor_uv_indices]
        
        # Floor XY
        floor_xy = np_coor2xy(floor_uv_img, z=z1, coorW=W, coorH=H, floorW=W, floorH=W)
        
        # Correct center
        center_offset_x = W / 2 - 0.5
        center_offset_y = W / 2 - 0.5
        floor_xy[:, 0] -= center_offset_x
        floor_xy[:, 1] -= center_offset_y
        
        # Draw on Panorama
        colors = cm.jet(np.linspace(0, 1, len(floor_xy)))
        
        ax1.scatter(floor_uv_img[:, 0], floor_uv_img[:, 1], c=colors, s=1)
        ceil_uv_indices = range(0, len(uv), 2)
        ceil_uv_img = uv_img[ceil_uv_indices]
        ax1.scatter(ceil_uv_img[:, 0], ceil_uv_img[:, 1], c=colors, s=1)
        
        # Ax2: Draw Floor Plan
        ax2.scatter(floor_xy[:, 0], floor_xy[:, 1], c=colors, s=10)
        
        # Add Camera (at origin)
        ax2.scatter([0], [0], c='black', marker='x', label='Camera')

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved visualization to {args.out}")

if __name__ == "__main__":
    args = parse_args()
    visualize_combined(args)
