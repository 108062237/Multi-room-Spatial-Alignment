
import json
import numpy as np
import sys
import os
from sklearn.decomposition import PCA

from utils.post_proc import np_coor2xy, np_xy2coor

def rectify_json(json_path, out_path):
    print(f"Rectifying {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    uv = np.array(data['uv'])
    z0 = data.get('z0', 50)
    z1 = data.get('z1', -50) # Use actual z1 from file if possible, or assume floor
    
    # We need to make sure we use the correct Z. 
    # The input file 08ddc172 has z1 approx -48.62
    z1 = data.get('z1')
    
    W = 1024
    H = 512
    floorW = 1024
    floorH = 1024
    
    # Extract Floor Points (odd indices)
    # The file should have 4 groups (8 points)
    if len(uv) != 8:
        print(f"Warning: Expected 8 points (4 groups), found {len(uv)}. Proceeding anyway.")
        
    floor_indices = range(1, len(uv), 2)
    uv_floor = uv[floor_indices].copy()
    
    # Scale to Pixel
    uv_floor_px = uv_floor.copy()
    uv_floor_px[:, 0] *= W
    uv_floor_px[:, 1] *= H
    
    # Convert to XY
    xy = np_coor2xy(uv_floor_px, z=z1, coorW=W, coorH=H, floorW=floorW, floorH=floorH)
    
    print("Original XY:")
    print(xy)
    
    # 1. Find dominant rotation using PCA
    pca = PCA(n_components=2)
    pca.fit(xy)
    
    # Components are the axes. components_[0] is the primary axis (longest variance)
    # Angle of primary axis
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    print(f"Detected rotation angle: {np.degrees(angle):.2f} degrees")
    
    # Rotation Matrix to align with X/Y
    # Rotate by -angle
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array(((c, -s), (s, c)))
    
    xy_rotated = xy.dot(R.T)
    
    # 2. Find Bounding Box in rotated space
    min_x = np.min(xy_rotated[:, 0])
    max_x = np.max(xy_rotated[:, 0])
    min_y = np.min(xy_rotated[:, 1])
    max_y = np.max(xy_rotated[:, 1])
    
    print(f"Bounding Box: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")
    
    # 3. Construct 4 corners of the Bounding Box
    # We need to order them to match the original winding or closest points
    # Let's define the 4 corners: (min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)
    # But we need to map them to the original indices to keep segment order?
    # Actually, HorizonNet polygon order usually matters.
    # Let's just create the 4 corners and match them to the nearest original rotated point to preserve order.
    
    rect_corners = [
        np.array([min_x, min_y]),
        np.array([min_x, max_y]),
        np.array([max_x, max_y]),
        np.array([max_x, min_y])
    ]
    
    rect_corners = np.array(rect_corners)
    
    # Match order
    new_xy_rotated = np.zeros_like(xy_rotated)
    
    for i in range(len(xy_rotated)):
        # Find closest rect corner
        orig = xy_rotated[i]
        dists = np.linalg.norm(rect_corners - orig, axis=1)
        best_idx = np.argmin(dists)
        new_xy_rotated[i] = rect_corners[best_idx]
        
    # 4. Rotate back
    # Rotate by +angle (inverse of R is R.T)
    R_inv = R.T # (c, s; -s, c) if R was rotation matrix. 
    # Wait, R was "rotate points by -angle". To reverse we rotate by +angle.
    # R_inv = np.array(((cos(a), -sin(a)), (sin(a), cos(a))))
    # Actually R defined above is:
    # [c -s]
    # [s  c]
    # where angle is -theta.
    # To invert, we use R^-1 = R^T.
    
    # xy_rotated = xy @ R.T
    # so xy_new = xy_rotated @ (R.T)^-1 = xy_rotated @ R
    
    xy_rectified = new_xy_rotated.dot(R)
    
    print("Rectified XY:")
    print(xy_rectified)
    
    # 5. Convert back to UV
    # Need Ceiling (z0) and Floor (z1)
    
    new_uv_list = []
    
    # For each group
    for i in range(len(xy_rectified)):
        p = xy_rectified[i].reshape(1, 2)
        
        # Ceiling
        coor_c = np_xy2coor(p, z=z0, coorW=W, coorH=H, floorW=floorW, floorH=floorH)
        u_c, v_c = coor_c[0]
        
        # Floor
        coor_f = np_xy2coor(p, z=z1, coorW=W, coorH=H, floorW=floorW, floorH=floorH)
        u_f, v_f = coor_f[0]
        
        # Append normalized
        new_uv_list.append([u_c/W, v_c/H])
        new_uv_list.append([u_f/W, v_f/H])
        
    data['uv'] = new_uv_list
    
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    target_json = "/home/wang/Desktop/Project/Project_result/08ddc172-5801-4ceb-8960-a61d2485d685/08ddc172-5801-4ceb-8960-a61d2485d685_adjusted.json"
    out_json = "/home/wang/Desktop/Project/Project_result/08ddc172-5801-4ceb-8960-a61d2485d685/08ddc172-5801-4ceb-8960-a61d2485d685_rectified.json"
    
    rectify_json(target_json, out_json)
