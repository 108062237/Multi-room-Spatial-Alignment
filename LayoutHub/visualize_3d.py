import json
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.signal import correlate2d
from scipy.ndimage import shift
import argparse
import sys
import os

# ==========================================
# Helper Functions (from HorizonNet/misc/panostretch.py & post_proc.py)
# ==========================================

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi

def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi

def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = -c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi

def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi

def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5

def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs, h)

    return np.stack([coorxs, coorys], axis=-1)


def cor_2_1d(cor, H, W):
    bon = np.zeros((2, W))
    n_cor = len(cor)
    
    # Ceiling
    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2],
                                  cor[(i*2+2) % n_cor],
                                  z=-50, w=W, h=H)
        xs = xys[:, 0].astype(int) % W
        ys = xys[:, 1]
        bon[0, xs] = ys

    # Floor
    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2+1],
                                  cor[(i*2+3) % n_cor],
                                  z=50, w=W, h=H)
        xs = xys[:, 0].astype(int) % W
        ys = xys[:, 1]
        bon[1, xs] = ys
        
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon

def layout_2_depth(cor_id, h, w, return_mask=False):
    # Convert corners to per-column boundary first
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    floor_h = 1.6
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth

# ==========================================
# Layout Parsing (Unified)
# ==========================================

def parse_layout(layout_path, img_w, img_h):
    """
    Parses layout file (JSON or TXT) and returns corner coordinates.
    Supports: HorizonNet, DMH-Net, LGT-Net, HoHoNet, Ground Truth.
    Returns:
        cor_id: numpy array of shape (N, 2) with pixel coordinates (x, y).
    """
    ext = os.path.splitext(layout_path)[1].lower()
    
    if ext == '.json':
        with open(layout_path, 'r') as f:
            data = json.load(f)
        
        if 'uv' in data:
            # HorizonNet / DMH-Net format
            uv_norm = np.array(data['uv'], dtype=np.float32) # Shape (N, 2), normalized 0-1
            
            # Convert normalized UV to pixel coordinates
            cor_id = np.zeros_like(uv_norm)
            cor_id[:, 0] = uv_norm[:, 0] * img_w
            cor_id[:, 1] = uv_norm[:, 1] * img_h
            return cor_id # Removed sort_corners_clockwise
            
        elif 'layoutPoints' in data:
            # LGT-Net format
            camera_height = data.get('cameraHeight', 1.6)
            camera_ceiling_height = data.get('cameraCeilingHeight', 1.6) # Default to symmetric if missing
            
            points = data['layoutPoints']['points']
            points.sort(key=lambda p: p['id'])
            
            # Check for VP file to handle rotation
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
                except Exception as e:
                    print(f"Failed to load VP from {vp_path}: {e}")
            
            cor_list = []
            
            for p in points:
                lgt_x_json, lgt_y_json, lgt_z_json = p['xyz']
                
                # 1. Convert to Internal Coords
                lgt_x = lgt_x_json
                lgt_y = lgt_y_json
                lgt_z = -lgt_z_json
                
                # 2. Apply Inverse Rotation if VP exists
                vec = np.array([lgt_x, lgt_y, lgt_z])
                if vp is not None:
                    vec = vec @ vp
                    lgt_x, lgt_y, lgt_z = vec
                
                # 3. Map to Panorama Coords (HorizonNet convention)
                pano_x_floor = lgt_x
                pano_y_floor = lgt_z
                pano_z_floor = -lgt_y 
                
                # Ceiling Point Calculation
                lgt_x_internal = lgt_x_json
                lgt_z_internal = -lgt_z_json
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
                    normXYZ = np.sqrt(x**2 + y**2 + z**2)
                    v = np.arcsin(z / normXYZ) if normXYZ > 0 else 0
                    u = np.arctan2(x, y) 
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
                
            return np.array(cor_list) # Removed sort_corners_clockwise
            
        else:
            raise ValueError(f"Unknown JSON format in {layout_path}")
            
    elif ext == '.txt':
        # HoHoNet/GT format: pixel coordinates x y per line
        try:
            cor_id = np.loadtxt(layout_path).astype(np.float32)
            
            # HoHoNet usually outputs in 1024x512, but let's check if we need scaling
            # If the coordinates are within the image range, we assume they are correct.
            # If they seem to be normalized (0-1), we scale them.
            if cor_id.max() <= 1.0:
                cor_id[:, 0] *= img_w
                cor_id[:, 1] *= img_h
            else:
                # Assume 1024x512 base resolution for HorizonNet TXT if not normalized
                # If the image resolution is different, we must scale the coordinates.
                # We use a heuristic: if the image is NOT 1024x512, we assume the layout is 1024x512 and scale it.
                if img_w != 1024 or img_h != 512:
                    print(f"Notice: Rescaling layout from 1024x512 to {img_w}x{img_h}")
                    cor_id[:, 0] = cor_id[:, 0] / 1024 * img_w
                    cor_id[:, 1] = cor_id[:, 1] / 512 * img_h
            
            return cor_id # Removed sort_corners_clockwise
        except Exception as e:
            raise ValueError(f"Failed to parse TXT layout {layout_path}: {e}")
    else:
        raise ValueError(f"Unsupported layout format: {ext}")

# ==========================================
# Main Viewer Logic
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', required=True,
                        help='Image texture in equirectangular format')
    parser.add_argument('--layout', required=True,
                        help='Txt or json file containing layout corners')
    parser.add_argument('--out', help='Output PLY file path (optional)')
    parser.add_argument('--vis', action='store_true', help='Show 3D visualization window')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',
                        help='Skip rendering ceiling')
    parser.add_argument('--ignore_wall', action='store_true',
                        help='Skip rendering wall')
    parser.add_argument('--ignore_wireframe', action='store_true',
                        help='Skip rendering wireframe')
    parser.add_argument('--no_texture', action='store_true',
                        help='Generate mesh without texture (white color)')
    parser.add_argument('--draw_outline', action='store_true',
                        help='Draw room outlines (boundaries) on the texture')
    args = parser.parse_args()

    # Reading source (texture img)
    if not os.path.exists(args.img):
        print(f"Error: Image not found at {args.img}")
        sys.exit(1)
        
    equirect_texture = np.array(Image.open(args.img))
    H, W = equirect_texture.shape[:2]

    # Parsing layout
    if not os.path.exists(args.layout):
        print(f"Error: Layout not found at {args.layout}")
        sys.exit(1)

    try:
        cor_id = parse_layout(args.layout, W, H)
    except Exception as e:
        print(f"Error parsing layout: {e}")
        sys.exit(1)

    # Resize for optimization
    scale_factor = 0.25
    import cv2
    new_W = int(W * scale_factor)
    new_H = int(H * scale_factor)
    equirect_texture = cv2.resize(equirect_texture, (new_W, new_H), interpolation=cv2.INTER_AREA)

    if args.no_texture:
        equirect_texture[:] = 255 # Set to white

    cor_id[:, 0] *= scale_factor
    cor_id[:, 1] *= scale_factor
    W, H = new_W, new_H

    if args.draw_outline:
        # Draw outlines on equirect_texture
        # Color: Black (0, 0, 0) if no_texture (white bg), else Green (0, 255, 0)
        line_color = (0, 0, 0) if args.no_texture else (0, 255, 0)
        thickness = 2
        
        N = len(cor_id) // 2
        for i in range(N):
            # Indices
            idx_c1 = 2 * i
            idx_f1 = 2 * i + 1
            idx_c2 = (2 * i + 2) % (2 * N)
            idx_f2 = (2 * i + 3) % (2 * N)
            
            # 1. Vertical Wall-Wall Boundary (c1 -> f1)
            pt1 = (int(cor_id[idx_c1][0]), int(cor_id[idx_c1][1]))
            pt2 = (int(cor_id[idx_f1][0]), int(cor_id[idx_f1][1]))
            cv2.line(equirect_texture, pt1, pt2, line_color, thickness)
            
            # 2. Ceiling Boundary (c1 -> c2)
            pts_c = pano_connect_points(cor_id[idx_c1], cor_id[idx_c2], z=-50, w=W, h=H)
            for x, y in pts_c:
                cv2.circle(equirect_texture, (int(x), int(y)), thickness // 2, line_color, -1)
                
            # 3. Floor Boundary (f1 -> f2)
            pts_f = pano_connect_points(cor_id[idx_f1], cor_id[idx_f2], z=50, w=W, h=H)
            for x, y in pts_f:
                cv2.circle(equirect_texture, (int(x), int(y)), thickness // 2, line_color, -1)

    # Convert corners to layout
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, return_mask=True)
    
    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth * np.sin(vs)
    cs = depth * np.cos(vs)
    xs = -cs * np.sin(us)
    ys = -cs * np.cos(us)

    # Aggregate mask
    mask = np.ones_like(floor_mask)
    if args.ignore_floor:
        mask &= ~floor_mask
    if args.ignore_ceiling:
        mask &= ~ceil_mask
    if args.ignore_wall:
        mask &= ~wall_mask

    # Prepare ply's points and faces
    xyzrgb = np.concatenate([
        xs[...,None], ys[...,None], zs[...,None],
        equirect_texture], -1)
    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:,[0]]], 1)
    mask = np.concatenate([mask, mask[:,[0]]], 1)
    lo_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1]])
    up_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 1]])
    ma_tri_template = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 0]])
    lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
    up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
    ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
    ref_mask = (
        lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
        up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
        ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack([
        ref_id[lo_mask],
        ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
    ], 1)
    faces_up_tri = np.stack([
        ref_id[up_mask],
        ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
        ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces_ma_tri = np.stack([
        ref_id[ma_mask],
        ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])

    # Dump results ply
    if args.out:
        # ply_header = '\n'.join([
        #     'ply',
        #     'format ascii 1.0',
        #     f'element vertex {len(points):d}',
        #     'property float x',
        #     'property float y',
        #     'property float z',
        #     'property uchar red',
        #     'property uchar green',
        #     'property uchar blue',
        #     f'element face {len(faces):d}',
        #     'property list uchar int vertex_indices',
        #     'end_header',
        # ])
        # with open(args.out, 'w') as f:
        #     f.write(ply_header)
        #     f.write('\n')
        #     for x, y, z, r, g, b in points:
        #         f.write(f'{x:.2f} {y:.2f} {z:.2f} {r:.0f} {g:.0f} {b:.0f}\n')
        #     for i, j, k in faces:
        #         f.write(f'3 {i:d} {j:d} {k:d}\n')
        # print(f"Saved PLY to {args.out}")
        mesh_out = o3d.geometry.TriangleMesh()
        mesh_out.vertices = o3d.utility.Vector3dVector(points[:, :3])
        mesh_out.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
        mesh_out.triangles = o3d.utility.Vector3iVector(faces)
        
        # 使用 Open3D 存檔，並指定 write_ascii=False (存成二進位)
        o3d.io.write_triangle_mesh(args.out, mesh_out, write_ascii=False)
        print(f"Saved optimized binary PLY to {args.out}")
    # Visualization
    if args.vis:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points[:, :3])
        mesh.vertex_colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        draw_geometries = [mesh]

        # Show wireframe
        if not args.ignore_wireframe:
            # Convert cor_id to 3d xyz
            N = len(cor_id) // 2
            floor_z = -1.6
            floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
            c = np.sqrt((floor_xy**2).sum(1))
            v = np_coory2v(cor_id[0::2, 1], H)
            ceil_z = (c * np.tan(v)).mean()

            # Prepare wireframe in open3d
            wf_points = [[x, y, floor_z] for x, y in floor_xy] +\
                        [[x, y, ceil_z] for x, y in floor_xy]
            wf_lines = [[i, (i+1)%N] for i in range(N)] +\
                       [[i+N, (i+1)%N+N] for i in range(N)] +\
                       [[i, i+N] for i in range(N)]
            wf_colors = [[1, 0, 0] for i in range(len(wf_lines))]
            wf_line_set = o3d.geometry.LineSet()
            wf_line_set.points = o3d.utility.Vector3dVector(wf_points)
            wf_line_set.lines = o3d.utility.Vector2iVector(wf_lines)
            wf_line_set.colors = o3d.utility.Vector3dVector(wf_colors)
            draw_geometries.append(wf_line_set)

        print("Opening 3D viewer...")
        o3d.visualization.draw_geometries(draw_geometries, mesh_show_back_face=True)
