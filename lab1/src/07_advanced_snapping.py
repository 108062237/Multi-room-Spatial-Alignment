#!/usr/bin/env python3
"""
Step 7: Advanced Floorplan Snapping (Corner + Wall)

This script maps polygons to global coordinates, ensures they are Axis-Aligned,
and then applies a rigorous Union-Find snapping logic:
1. Corner Snapping: If two corners are within D (e.g. 0.5m) Euclidean distance, 
   their X and Y walls are explicitly merged.
2. Wall Snapping: If two parallel walls are within W (e.g. 0.2m) distance and overlap, 
   they are merged.

Usage:
  python src/07_advanced_snapping.py \
    --scene_dir data/group/58472_Floor1 \
    --poses     data/group/58472_Floor1/perfect_optimized_poses.json \
    --out       data/group/58472_Floor1/viz/advanced_snapped.png
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional

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
from src.utils.geom import se2_apply, rectify_polygon
from src.utils.labels import get_room_labels, get_display_label

import importlib.util

_LAYOUTHUB = Path(__file__).resolve().parent.parent.parent / 'LayoutHub'
_POST_PROC_PY = _LAYOUTHUB / 'utils' / 'post_proc.py'
try:
    spec = importlib.util.spec_from_file_location("post_proc", str(_POST_PROC_PY))
    post_proc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_proc)
    _np_coor2xy = post_proc.np_coor2xy
    _HAS_COOR2XY = True
except Exception:
    _HAS_COOR2XY = False

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def load_layout_gt_txt_as_local_xy(txt_path: Path, W: int, H: int) -> Optional[np.ndarray]:
    if not txt_path.exists() or not _HAS_COOR2XY:
        return None
    pts = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])
    if len(pts) < 2: return None
    if len(pts) % 2 == 0:
        floor_pixel = np.array([pts[i] if pts[i][1] > pts[i+1][1] else pts[i+1] for i in range(0, len(pts), 2)])
    else:
        floor_pixel = np.array(pts)
    if len(floor_pixel) < 3: return None
    
    floor_xy = _np_coor2xy(floor_pixel, z=50, coorW=W, coorH=H, floorW=W, floorH=W)
    center = W / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]
    
    return rectify_polygon(floor_xy).astype(np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matches", type=str, default="", help="If provided, use exact explicit corner match bindings instead of heuristic distances.")
    ap.add_argument("--corner_threshold", type=float, default=0.5, help="Euclidean distance for corner snapping")
    ap.add_argument("--wall_threshold", type=float, default=0.2, help="1D distance for wall snapping")
    ap.add_argument("--pano_w", type=int, default=1024)
    ap.add_argument("--pano_h", type=int, default=512)
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    manifest = json.loads((scene_dir / "manifest.json").read_text())
    nodes = manifest.get("nodes", [])
    label_map = get_room_labels(scene_dir)
    poses = json.loads(Path(args.poses).read_text()).get("poses", {})

    polys_world = []
    cam_xy = []
    pids = []

    for n in nodes:
        pid = n.get("pano_id", "")
        if not pid or pid not in poses: continue
        lp_path = scene_dir / "layout_gt" / f"{pid}.txt"
        if not lp_path.exists(): continue
        poly_local = load_layout_gt_txt_as_local_xy(lp_path, W=args.pano_w, H=args.pano_h)
        if poly_local is None: continue

        x, y, th = float(poses[pid]["x"]), float(poses[pid]["y"]), float(poses[pid]["theta"])
        poly_world_noisy = se2_apply((x, y, th), poly_local)
        
        centroid = np.mean(poly_world_noisy, axis=0)
        centered = poly_world_noisy - centroid
        poly_world = rectify_polygon(centered, rotate_back=False) + centroid
        
        polys_world.append(poly_world)
        cam_xy.append([x, y])
        pids.append(pid)

    if not polys_world:
        print("No valid polygons!")
        return

    # Extract X/Y Walls and Corners
    # A wall is represented by a unique ID
    # x_walls[id] = {'val': X, 'nodes': [(poly_idx, p1_idx, p2_idx)]}
    class Wall:
        def __init__(self, val):
            self.val = val
            self.pts = [] # lists of (poly_idx, point_idx)

    x_walls = []
    y_walls = []
    
    # Corner to its X_wall and Y_wall index
    corner_to_walls = [] 
    
    for i, poly in enumerate(polys_world):
        N = len(poly)
        corner_walls = []
        for j in range(N):
            corner_walls.append({'x': -1, 'y': -1, 'coord': poly[j]})
        corner_to_walls.append(corner_walls)

    for i, poly in enumerate(polys_world):
        N = len(poly)
        for j in range(N):
            p1, p2 = j, (j+1)%N
            if abs(poly[p1][0] - poly[p2][0]) < abs(poly[p1][1] - poly[p2][1]):
                # Vertical -> X wall
                wid = len(x_walls)
                x_walls.append(Wall(poly[p1][0]))
                x_walls[-1].pts.extend([(i, p1), (i, p2)])
                corner_to_walls[i][p1]['x'] = wid
                corner_to_walls[i][p2]['x'] = wid
            else:
                # Horizontal -> Y wall
                wid = len(y_walls)
                y_walls.append(Wall(poly[p1][1]))
                y_walls[-1].pts.extend([(i, p1), (i, p2)])
                corner_to_walls[i][p1]['y'] = wid
                corner_to_walls[i][p2]['y'] = wid

    uf_x = UnionFind(len(x_walls))
    uf_y = UnionFind(len(y_walls))

    # Phase 1: True Corner Snapping OR Explicit Matches Snapping
    snapped_count = 0
    if args.matches and Path(args.matches).exists():
        matches = json.loads(Path(args.matches).read_text())
        for match in matches:
            src_pid = match["src"].replace(".txt", "")
            dst_pid = match["dst"].replace(".txt", "")
            if src_pid not in pids or dst_pid not in pids: continue
            
            src_i = pids.index(src_pid)
            dst_i = pids.index(dst_pid)
            
            for k in range(len(match["idx_src"])):
                # idx in matches is 1-indexed
                p_src = match["idx_src"][k] - 1
                p_dst = match["idx_dst"][k] - 1
                
                c_src = corner_to_walls[src_i][p_src]
                c_dst = corner_to_walls[dst_i][p_dst]
                
                if c_src['x'] != -1 and c_dst['x'] != -1:
                    uf_x.union(c_src['x'], c_dst['x'])
                    snapped_count += 1
                if c_src['y'] != -1 and c_dst['y'] != -1:
                    uf_y.union(c_src['y'], c_dst['y'])
                    snapped_count += 1
                    
        print(f"[*] 套用顯式配對: 成功綁定了 {snapped_count} 條拓樸邊界 (無視物理誤差)")
    elif args.corner_threshold > 0:
        corners = [] # (x, y, room, idx)
        for i, c_list in enumerate(corner_to_walls):
            for j, c in enumerate(c_list):
                # 只有同時存在垂直牆(X)與水平牆(Y)的點，才是真實的 90 度房間轉角 (排除門上的平直線控制點)
                if c['x'] != -1 and c['y'] != -1:
                    corners.append((c['coord'][0], c['coord'][1], i, j))
        
        snapped_count = 0
        for k1 in range(len(corners)):
            for k2 in range(k1+1, len(corners)):
                c1, c2 = corners[k1], corners[k2]
                if c1[2] == c2[2]: continue # 不找同一個房間的角點
                dist = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
                if dist < args.corner_threshold:
                    w1_x = corner_to_walls[c1[2]][c1[3]]['x']
                    w2_x = corner_to_walls[c2[2]][c2[3]]['x']
                    uf_x.union(w1_x, w2_x)
                    
                    w1_y = corner_to_walls[c1[2]][c1[3]]['y']
                    w2_y = corner_to_walls[c2[2]][c2[3]]['y']
                    uf_y.union(w1_y, w2_y)
                    snapped_count += 1
        print(f"[*] 成功配對並強力吸附了 {snapped_count} 組合適的相近角點 (閾值 {args.corner_threshold}m)")

    # Phase 2: Wall Snapping based on 1D distance
    def snap_walls_1d(walls, uf):
        srt = sorted(range(len(walls)), key=lambda idx: walls[idx].val)
        curr_cluster = [srt[0]]
        for idx in srt[1:]:
            # If wall is within distance threshold of the first item in current cluster
            if abs(walls[idx].val - walls[curr_cluster[0]].val) <= args.wall_threshold:
                curr_cluster.append(idx)
            else:
                for c_idx in curr_cluster[1:]:
                    uf.union(curr_cluster[0], c_idx)
                curr_cluster = [idx]
        for c_idx in curr_cluster[1:]:
            uf.union(curr_cluster[0], c_idx)

    if args.wall_threshold > 0:
        snap_walls_1d(x_walls, uf_x)
        snap_walls_1d(y_walls, uf_y)

    # Phase 3: Calculate Mean for each group and Assign
    def apply_uf_means(walls, uf, axis):
        groups = {}
        for i in range(len(walls)):
            root = uf.find(i)
            groups.setdefault(root, []).append(i)
        
        for root, indices in groups.items():
            mean_val = np.mean([walls[idx].val for idx in indices])
            for idx in indices:
                for poly_idx, p_idx in walls[idx].pts:
                    polys_world[poly_idx][p_idx][axis] = mean_val

    apply_uf_means(x_walls, uf_x, 0)
    apply_uf_means(y_walls, uf_y, 1)

    # Drawing
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
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.25, zorder=2)
        cx, cy = cam_xy[i]
        label_text = get_display_label(pids[i], label_map)
        ax.text(cx, cy - 0.1, label_text, fontsize=9, ha='center', va='top', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1), zorder=10)

    ax.scatter([c[0] for c in cam_xy], [c[1] for c in cam_xy], s=25, c='black', zorder=5)

    ax.set_title(f"{scene_dir.name} | Advanced Mapping\nCorner Snapped: <= {args.corner_threshold}m, Wall Snapped: <= {args.wall_threshold}m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)

    all_pts = np.vstack([p for p in polys_world])
    xmin, ymin = np.min(all_pts, axis=0)
    xmax, ymax = np.max(all_pts, axis=0)
    pad = 0.1 * max(xmax - xmin, ymax - ymin, 1e-6)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    plt.close(fig)

    print(f"[OK] Wrote advanced floorplan to {args.out}")

if __name__ == "__main__":
    main()
