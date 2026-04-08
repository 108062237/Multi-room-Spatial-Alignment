import os
import sys
import json
import argparse
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# Import necessary functions from utils and existing scripts
from utils.post_proc import np_coor2xy
from utils.panorama import lineIdxFromCors

from visualize_corners import parse_layout

# Max display width for the panorama panel; prevents huge memory usage for 8192px images
PANO_DISPLAY_W = 2048

def get_color(idx, total, mode='hsv'):
    """Generate a color for index i out of total."""
    x = idx / max(1, total)
    c = cm.hsv(x)
    return c

def manhattan_align(floor_xy):
    """
    Rotate floor_xy so that wall edges align to the X/Y axes.
    Strategy: collect all edge angles, fold them into [0, 90°) so opposite
    walls map to the same angle, then take the length-weighted mean.
    Rotate by (mean_angle mod 90°) so the dominant wall direction maps to
    the nearest axis.
    Returns rotated floor_xy and the 2x2 rotation matrix.
    """
    n = len(floor_xy)
    angles = []
    weights = []
    for i in range(n):
        p1 = floor_xy[i]
        p2 = floor_xy[(i + 1) % n]
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue
        angle = np.arctan2(d[1], d[0])            # in (-π, π]
        angle_deg = np.degrees(angle) % 180        # fold into [0, 180)
        angle_deg = angle_deg % 90                 # fold into [0, 90)
        angles.append(angle_deg)
        weights.append(length)

    if len(angles) == 0:
        return floor_xy, np.eye(2)

    angles = np.array(angles)
    weights = np.array(weights)

    # Weighted circular mean inside [0, 90)
    sin_vals = np.sin(np.radians(2 * angles))  # double angle trick for 90° period
    cos_vals = np.cos(np.radians(2 * angles))
    mean_angle = np.degrees(np.arctan2(
        np.average(sin_vals, weights=weights),
        np.average(cos_vals, weights=weights)
    )) / 2  # halve back

    # mean_angle is now in [0, 90): how much the dominant wall is off from 0°
    rot_angle = -np.radians(mean_angle)
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    R = np.array([[c, -s], [s, c]])
    aligned = floor_xy @ R.T
    return aligned, R

def visualize_combined_corners(layout_path, image_path, output_path):
    print(f"Processing {layout_path} ...")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img_src = cv2.imread(image_path)
    if img_src is None:
        print(f"Failed to read image: {image_path}")
        return

    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    H, W = img_src.shape[:2]
    print(f"  Image size: {W}x{H}")

    # 1. Parse Layout (returns pixel coords scaled to actual W, H)
    cor_id = parse_layout(layout_path, W, H)
    if cor_id is None:
        print(f"Failed to parse layout: {layout_path}")
        return

    num_points = len(cor_id)
    num_corners = num_points // 2

    # ----------------------------------------------------------------
    # For very large panoramas, downsample the image for display only.
    # All coordinate calculations still use the original W, H.
    # ----------------------------------------------------------------
    if W > PANO_DISPLAY_W:
        scale = PANO_DISPLAY_W / W
        display_H = int(H * scale)
        img_display = cv2.resize(img_src, (PANO_DISPLAY_W, display_H))
        # Scale cor_id for display
        cor_id_display = cor_id.copy().astype(float)
        cor_id_display[:, 0] *= scale
        cor_id_display[:, 1] *= scale
        dW, dH = PANO_DISPLAY_W, display_H
    else:
        img_display = img_src
        cor_id_display = cor_id.copy().astype(float)
        dW, dH = W, H

    # 2. Project floor corners to top-down XY using original resolution
    floor_cor_id = cor_id[1::2]   # odd rows = floor points
    z1 = 50  # floor depth (pixels in HorizonNet convention)

    floor_xy = np_coor2xy(floor_cor_id, z=z1, coorW=W, coorH=H, floorW=W, floorH=W)

    # Shift so camera is at origin
    center_offset_x = W / 2 - 0.5
    center_offset_y = W / 2 - 0.5
    floor_xy[:, 0] -= center_offset_x
    floor_xy[:, 1] -= center_offset_y

    # 3. Manhattan alignment – rotate floor plan to align walls with XY axes
    floor_xy_aligned, R = manhattan_align(floor_xy)

    # Flip Y so corner ordering is clockwise (1→2→3→4) in the top-down view
    # (np_coor2xy produces Y increasing away from camera, which reverses CW/CCW)
    floor_xy_aligned[:, 1] = -floor_xy_aligned[:, 1]

    # 4. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # ----------------------------------------------------------------
    # Left panel: Panorama View
    # Use lineIdxFromCors (same as visualize_corners.py) for each wall
    # segment so the geodesic lines are geometrically correct.
    # Build a color-painted image, then overlay number labels via matplotlib.
    # ----------------------------------------------------------------

    # Start from a copy of the display image
    pano_drawn = img_display.copy().astype(np.uint8)

    for i in range(num_corners):
        c_idx = 2 * i
        f_idx = 2 * i + 1
        next_i = (i + 1) % num_corners
        next_c_idx = 2 * next_i
        next_f_idx = 2 * next_i + 1

        color_float = get_color(i, num_corners)        # RGBA 0-1
        color_u8 = tuple(int(c * 255) for c in color_float[:3])  # RGB 0-255

        # Build a 4-point segment:
        # ceiling[i] -> ceiling[next], floor[i] -> floor[next]
        # plus vertical: ceiling[i]<->floor[i]
        # We use the same structure as draw_boundary_from_cor_id but only
        # for the two points of this segment.
        seg_corners = np.array([
            cor_id_display[c_idx],   # ceiling i
            cor_id_display[f_idx],   # floor   i
            cor_id_display[next_c_idx],  # ceiling next
            cor_id_display[next_f_idx],  # floor   next
        ])

        # Lines to draw (as point pairs for lineIdxFromCors):
        # ceiling_i  <-> ceiling_next   (horizontal ceiling edge)
        # floor_i    <-> floor_next     (horizontal floor edge)
        # ceiling_i  <-> floor_i        (vertical corner line)
        line_pairs = np.array([
            [cor_id_display[c_idx],      cor_id_display[next_c_idx]],
            [cor_id_display[f_idx],      cor_id_display[next_f_idx]],
            [cor_id_display[c_idx],      cor_id_display[f_idx]],
        ])  # shape (3, 2, 2)

        # Flatten to alternating pairs expected by lineIdxFromCors
        pts_flat = line_pairs.reshape(-1, 2)  # (6, 2)

        try:
            rs, cs = lineIdxFromCors(pts_flat, dW, dH)
            rs = np.clip(np.array(rs), 0, dH - 1)
            cs = np.clip(np.array(cs), 0, dW - 1)

            # Create mask and dilate for thickness
            mask = np.zeros((dH, dW), dtype=np.uint8)
            mask[rs, cs] = 255
            thick = max(3, int(dW / 512))  # scale thickness with image width
            kernel = np.ones((thick, thick), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            pano_drawn[mask > 0] = color_u8
        except Exception as e:
            print(f"  Warning: could not draw segment {i}: {e}")

    ax1.imshow(pano_drawn)
    ax1.set_title("2D Layout (Panorama)", fontsize=16)
    ax1.axis('off')

    # Overlay corner number labels on top (at floor corner positions)
    label_pad = max(10, int(dH * 0.015))
    for i in range(num_corners):
        f_idx = 2 * i + 1
        px, py = cor_id_display[f_idx]
        color_float = get_color(i, num_corners)
        label_text = str(i + 1)
        ax1.text(px, py - label_pad, label_text,
                 color='white', fontsize=14, fontweight='bold',
                 ha='center', va='bottom',
                 bbox=dict(facecolor=color_float[:3], alpha=0.85,
                           edgecolor='none', boxstyle='round,pad=0.3'))

    # ----------------------------------------------------------------
    # Right panel: Top-Down Floor Plan (Manhattan aligned)
    # ----------------------------------------------------------------
    ax2.set_title("Top-Down Floor Plan (Manhattan Aligned)", fontsize=16)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax2.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    # Camera at origin (also rotated — stays at 0,0)
    ax2.scatter([0], [0], marker='^', color='k', s=200, zorder=5, label='Camera')

    label_scale = (floor_xy_aligned.max() - floor_xy_aligned.min()) * 0.04
    label_scale = max(label_scale, 1.0)

    for i in range(num_corners):
        next_i = (i + 1) % num_corners
        color = get_color(i, num_corners)

        fx1, fy1 = floor_xy_aligned[i]
        fx2, fy2 = floor_xy_aligned[next_i]

        ax2.plot([fx1, fx2], [fy1, fy2], color=color, linewidth=3)
        ax2.scatter([fx1], [fy1], color=color, s=120, marker='o',
                    edgecolors='black', linewidths=1, zorder=4)

        # Label offset: push away from camera center
        norm = np.sqrt(fx1 ** 2 + fy1 ** 2)
        if norm > 1e-6:
            ox = (fx1 / norm) * label_scale
            oy = (fy1 / norm) * label_scale
        else:
            ox, oy = label_scale, label_scale

        ax2.text(fx1 + ox, fy1 + oy, str(i + 1),
                 color='black', fontsize=13, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.85,
                           edgecolor=color, linewidth=1.5,
                           boxstyle='round,pad=0.3'))

    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined visualization to {output_path}")

def process_single_pair(layout_path, image_path, output_path):
    if not layout_path or not image_path or not output_path:
        print("Error: --layout, --image, and --out are required for single file mode.")
        return
    visualize_combined_corners(layout_path, image_path, output_path)

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
        file_root = os.path.splitext(basename)[0]
        
        candidates = [
            file_root + ".jpg",
            file_root + ".png",
            file_root + ".raw.png",
            basename.replace(".json", ".raw.png")
        ]
        
        img_path = None
        for cand in candidates:
            cand_path = os.path.join(image_dir, cand)
            if os.path.exists(cand_path):
                img_path = cand_path
                break
        
        if img_path is None:
            img_path = os.path.join(image_dir, file_root + ".jpg")
        
        output_filename = file_root + "_viz_combined.png"
        output_path = os.path.join(output_dir, output_filename)
        
        visualize_combined_corners(layout_file, img_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Visualize Combined Layout with Corner Labels")
    
    parser.add_argument("--layout", help="Path to a single layout file (JSON or TXT)")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--out", help="Path to output image file")
    
    parser.add_argument("--layout_dir", help="Directory containing layout JSON/TXT files")
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
