#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path
import argparse
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.geom import pano_xy_to_u_v, ray_from_uv, intersect_with_z_plane


def load_layout_txt_local_xy(layout_txt: Path, W=1024, H=512, z_floor=-1.0):
    """
    Parses Floor Pixels and converts them to 3D local XY coordinates.
    """
    if not layout_txt.exists():
        return None

    lines = []
    for line in layout_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            lines.append((float(parts[0]), float(parts[1])))

    if len(lines) < 6:
        return None
    if len(lines) % 2 != 0:
        lines = lines[:-1]

    floor_pixels = []
    for i in range(0, len(lines), 2):
        p1 = lines[i]
        p2 = lines[i+1]
        # Max Y corresponds to the floor (bottom of image)
        if p1[1] > p2[1]:
            floor_pixels.append(p1)
        else:
            floor_pixels.append(p2)
        
    pts_xy = []
    for (px, py) in floor_pixels:
        u, v = pano_xy_to_u_v(px, py, W, H)
        d = ray_from_uv(u, v)
        p3 = intersect_with_z_plane(d, z_plane=z_floor)
        if p3 is None:
            continue
        pts_xy.append([p3[0], p3[1]])

    if len(pts_xy) < 3:
        return None

    return np.array(pts_xy, dtype=np.float64)



def estimate_theta_from_layout(layout_txt: Path, W=1024, H=512):
    """
    Estimates the rotation 'theta' such that the room aligns with Manhattan axes.
    Uses length-weighted circular mean for 4-fold symmetry (Manhattan World).
    """
    xy = load_layout_txt_local_xy(layout_txt, W=W, H=H) # (N,2)
    if xy is None or xy.shape[0] < 3:
        return 0.0

    N = xy.shape[0]
    
    # 用來累加方向向量的 X, Y 分量
    sum_cos = 0.0
    sum_sin = 0.0
    
    for i in range(N):
        p1 = xy[i]
        p2 = xy[(i + 1) % N]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        angle = math.atan2(dy, dx)
        length = math.hypot(dx, dy)
        
        # 核心數學：乘以 4 來抹平 0, 90, 180, 270 度的差異
        # 這樣一來，所有互相垂直的牆壁，都會指向同一個 2D 向量方向！
        sum_cos += length * math.cos(4.0 * angle)
        sum_sin += length * math.sin(4.0 * angle)
        
    # 算出綜合的 4 倍角
    dominant_angle_x4 = math.atan2(sum_sin, sum_cos)
    
    # 除以 4 換回真實角度。
    # 因為 atan2 的範圍是 [-pi, pi]，除以 4 後剛好完美落在 [-pi/4, pi/4]！
    dominant_angle = dominant_angle_x4 / 4.0
    
    # 我們需要的是「旋轉修正量」，也就是把 dominant_angle 轉回 0 的角度
    theta = -dominant_angle
        
    return float(theta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="data/group/<scene>/manifest.json")
    ap.add_argument("--out", required=True, help="output theta priors json")
    ap.add_argument("--W", type=int, default=1024)
    ap.add_argument("--H", type=int, default=512)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    scene_dir = manifest_path.parent
    
    manifest = json.loads(manifest_path.read_text())
    priors = {
        "scene_id": manifest.get("scene_id", ""),
        "theta_priors": {}
    }

    n_ok = 0
    n_fail = 0
    for node in manifest["nodes"]:
        pano_id = node["pano_id"]
        layout_path_str = node.get("layout_gt_path", "")
        
        # Robust Path Resolution
        if not layout_path_str:
            priors["theta_priors"][pano_id] = 0.0
            n_fail += 1
            continue
            
        layout_path = Path(layout_path_str)
        if not layout_path.is_absolute():
            layout_path = scene_dir / layout_path
            
        if not layout_path.exists():
            priors["theta_priors"][pano_id] = 0.0
            n_fail += 1
            continue
            
        th = estimate_theta_from_layout(layout_path, W=args.W, H=args.H)
        priors["theta_priors"][pano_id] = th
        n_ok += 1

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(priors, indent=2))
    print(f"[OK] wrote theta priors -> {args.out} (ok={n_ok}, fail={n_fail})")


if __name__ == "__main__":
    main()
