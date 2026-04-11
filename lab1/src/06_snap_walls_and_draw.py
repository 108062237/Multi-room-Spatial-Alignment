#!/usr/bin/env python3
"""
Step 6: Draw floorplan overlay with Global Wall Snapping.

Inputs:
  - manifest.json (contains nodes with layout_gt_path)
  - poses.json (initial_poses.json or optimized_poses.json)

Output:
  - snapped_floorplan_overlay.png

Usage:
  python src/06_snap_walls_and_draw.py \
    --scene_dir data/group/58472_Floor1 \
    --poses     data/group/58472_Floor1/optimized_poses.json \
    --out       data/group/58472_Floor1/viz/snapped_floorplan_overlay.png \
    --snap_threshold 0.2
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.geom import se2_compose, pano_xy_to_u_v, ray_from_uv, intersect_with_z_plane, rectify_polygon
from src.utils.labels import get_room_labels, get_display_label

import importlib.util

_LAYOUTHUB = Path(__file__).resolve().parent.parent.parent / 'LayoutHub'
_POST_PROC_PY = _LAYOUTHUB / 'utils' / 'post_proc.py'

try:
    if _POST_PROC_PY.exists():
        spec = importlib.util.spec_from_file_location("post_proc", str(_POST_PROC_PY))
        post_proc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(post_proc)
        _np_coor2xy = post_proc.np_coor2xy
        _HAS_COOR2XY = True
    else:
        print(f"\n找不到 LayoutHub 裡面的 Python 檔: {_POST_PROC_PY}\n")
        _HAS_COOR2XY = False
except Exception as e:
    print(f"\n強制讀取 post_proc.py 失敗，原因: {e}\n")
    _HAS_COOR2XY = False


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def se2_apply(pose: Tuple[float, float, float], pts: np.ndarray) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([x, y], dtype=np.float64)


def load_layout_gt_txt_as_local_xy(txt_path: Path, W: int, H: int, z_floor: float = -1.0) -> Optional[np.ndarray]:
    if not txt_path.exists():
        return None

    pts = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    if len(pts) < 2:
        return None

    if len(pts) % 2 == 0:
        floor_pixel = np.array(
            [pts[i] if pts[i][1] > pts[i+1][1] else pts[i+1]
             for i in range(0, len(pts), 2)]
        )
    else:
        floor_pixel = np.array(pts)

    if len(floor_pixel) < 3:
        return None

    if not _HAS_COOR2XY:
        return None

    pano_w = W
    z = 50
    floor_xy = _np_coor2xy(floor_pixel, z=z,
                           coorW=pano_w, coorH=H,
                           floorW=pano_w, floorH=pano_w)
    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]

    # 強制修正曼哈頓多邊形
    floor_xy = rectify_polygon(floor_xy)

    return floor_xy.astype(np.float64)

def snap_1d_walls(walls, threshold):
    walls.sort(key=lambda w: w["val"])
    if not walls: return
    clusters = []
    curr = [walls[0]]
    for w in walls[1:]:
        # 只要距離群組開頭在 threshold 內就算同一群 (避免 single-linkage 的連鎖效應)
        if abs(w["val"] - curr[0]["val"]) <= threshold:
            curr.append(w)
        else:
            clusters.append(curr)
            curr = [w]
    clusters.append(curr)
    
    for c in clusters:
        # 有多於一道牆才能吸附，如果你想單看厚度也可以在這偏移
        if len(c) > 1:
            mean_val = np.mean([w["val"] for w in c])
            for w in c:
                w["new_val"] = mean_val
        else:
            c[0]["new_val"] = c[0]["val"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True, help="data/group/<scene>")
    ap.add_argument("--poses", type=str, required=True, help="optimized_poses.json or initial_poses.json")
    ap.add_argument("--out", type=str, required=True, help="output image path (png)")
    ap.add_argument("--snap_threshold", type=float, default=0.2, help="吸附牆壁的閾值 (單位: 公尺)。例如 0.2 代表 20 公分內的水平或垂直牆會吸附")
    ap.add_argument("--alpha", type=float, default=0.25, help="polygon fill alpha")
    ap.add_argument("--draw_camera_points", action="store_true", help="draw camera points")
    ap.add_argument("--limit_nodes", type=int, default=0, help="debug: draw only first K nodes (0=all)")
    ap.add_argument("--pano_w", type=int, default=1024, help="Width of panorama image")
    ap.add_argument("--pano_h", type=int, default=512, help="Height of panorama image")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    manifest = load_json(scene_dir / "manifest.json")
    nodes = manifest.get("nodes", [])

    label_map = get_room_labels(scene_dir)

    poses_data = load_json(Path(args.poses))
    poses = poses_data.get("poses", {})
    if not poses:
        raise RuntimeError("poses json contains no poses")

    polys_world = []
    cam_xy = []
    pids = []
    doors = []
    drawn = 0
    skipped_no_layout = 0
    skipped_parse_fail = 0
    skipped_no_pose = 0

    iter_nodes = nodes
    if args.limit_nodes and args.limit_nodes > 0:
        iter_nodes = nodes[: args.limit_nodes]

    for n in iter_nodes:
        pid = n.get("pano_id", "")
        if not pid: continue
        if pid not in poses:
            skipped_no_pose += 1
            continue

        lp_path = scene_dir / "layout_gt" / f"{pid}.txt"
        if not lp_path.exists():
            skipped_no_layout += 1
            continue

        poly_local = load_layout_gt_txt_as_local_xy(lp_path, W=args.pano_w, H=args.pano_h)
        if poly_local is None:
            skipped_parse_fail += 1
            continue

        x = float(poses[pid]["x"])
        y = float(poses[pid]["y"])
        th = float(poses[pid]["theta"])
        
        # 1. Room boundaries mapping to world (it will have slight cumulative GTSAM rotation noise)
        poly_world_noisy = se2_apply((x, y, th), poly_local)
        
        # 2. Force Global Manhattan Axis-Alignment!
        # 這會消除 GTSAM theta 與 Local bias 造成的微弱非平行的旋轉，完美貼齊正交座標軸
        centroid = np.mean(poly_world_noisy, axis=0)
        centered = poly_world_noisy - centroid
        poly_world_perfect = rectify_polygon(centered, rotate_back=False)
        poly_world = poly_world_perfect + centroid

        for conn in n.get("connections", []):
            hx, hy = conn["hotspot_xy"]
            gx = x + math.cos(th) * hx - math.sin(th) * hy
            gy = y + math.sin(th) * hx + math.cos(th) * hy
            doors.append((gx, gy))

        polys_world.append(poly_world)
        cam_xy.append([x, y])
        pids.append(pid)
        drawn += 1

    if drawn == 0:
        raise RuntimeError("No layout polygons drawn.")

    cam_xy = np.array(cam_xy, dtype=np.float64) if cam_xy else np.zeros((0, 2))

    # ===============================================
    # 進行全域牆壁吸附 (Global Snapping)
    # ===============================================
    x_walls = []
    y_walls = []
    if args.snap_threshold > 0:
        for i, poly in enumerate(polys_world):
            N = len(poly)
            for j in range(N):
                p1 = poly[j]
                p2 = poly[(j+1)%N]
                if abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]):
                    x_walls.append({"val": p1[0], "poly_idx": i, "p1": j, "p2": (j+1)%N})
                else:
                    y_walls.append({"val": p1[1], "poly_idx": i, "p1": j, "p2": (j+1)%N})
                    
        snap_1d_walls(x_walls, args.snap_threshold)
        snap_1d_walls(y_walls, args.snap_threshold)
        
        # 覆寫回去
        for w in x_walls:
            polys_world[w["poly_idx"]][w["p1"], 0] = w["new_val"]
            polys_world[w["poly_idx"]][w["p2"], 0] = w["new_val"]
        for w in y_walls:
            polys_world[w["poly_idx"]][w["p1"], 1] = w["new_val"]
            polys_world[w["poly_idx"]][w["p2"], 1] = w["new_val"]
    # ===============================================

    out_path = Path(args.out)
    ensure_parent_dir(out_path)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    if HAS_SHAPELY and len(polys_world) > 1:
        shapely_polys = [Polygon(p).buffer(0) for p in polys_world]
        union_poly = unary_union(shapely_polys)
        
        if union_poly.geom_type == 'MultiPolygon':
            for geom in union_poly.geoms:
                ex_x, ex_y = geom.exterior.xy
                ax.plot(ex_x, ex_y, color='black', linewidth=3)
        else:
            ex_x, ex_y = union_poly.exterior.xy
            ax.plot(ex_x, ex_y, color='black', linewidth=3)


    for i, poly in enumerate(polys_world):
        closed = np.vstack([poly, poly[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], color='black', linewidth=1.5, alpha=0.8, zorder=3)
        ax.fill(poly[:, 0], poly[:, 1], alpha=args.alpha, zorder=2)
        
        cx, cy = cam_xy[i]
        label_text = get_display_label(pids[i], label_map)
        ax.text(cx, cy - 0.1, label_text, fontsize=9, ha='center', va='top', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1), zorder=10)

    if doors:
        dx = [d[0] for d in doors]
        dy = [d[1] for d in doors]
        ax.scatter(dx, dy, marker='s', color='#FF4500', edgecolor='black', s=45, zorder=6, label='Doors')

    if args.draw_camera_points and cam_xy.shape[0] > 0:
        ax.scatter(cam_xy[:, 0], cam_xy[:, 1], s=25, c='black', zorder=5)

    ax.set_title(
        f"{scene_dir.name} (Snapped) | polygons drawn={drawn}, gap_threshold={args.snap_threshold}m\n"
        f"skipped(no_pose={skipped_no_pose}, no_layout={skipped_no_layout}, parse_fail={skipped_parse_fail})"
    )
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)

    all_pts = np.vstack([p for p in polys_world])
    xmin, ymin = np.min(all_pts, axis=0)
    xmax, ymax = np.max(all_pts, axis=0)
    pad = 0.1 * max(xmax - xmin, ymax - ymin, 1e-6)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)

    print(f"[OK] wrote snapped floorplan overlay -> {out_path}")
    print(f"     polygons drawn={drawn}")

if __name__ == "__main__":
    main()
