#!/usr/bin/env python3
"""
Step 8: Perfect Tree Snapping (Acyclic Graph Translations + 1D Wall Snapping)

Bypasses GTSAM translation noises entirely by treating the map as a strict tree.
1. Aligns each room to local Manhattan frame.
2. Applies initial GTSAM pose just for topological sorting/rotation.
3. Rigidly translates rooms downwards through the BFS tree to EXACTLY align the centroid of matched door points.
4. Performs 1D wall snapping to erase any orthogonal micro-gaps.
"""

import argparse, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
spec = importlib.util.spec_from_file_location("post_proc", str(_LAYOUTHUB / 'utils' / 'post_proc.py'))
post_proc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post_proc)
_np_coor2xy = post_proc.np_coor2xy

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j: self.parent[root_i] = root_j

def load_layout(txt_path, W, H):
    pts = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        pts.append([float(x) for x in line.replace(",", " ").split()[:2]])
    if len(pts)%2 == 0:
        pts = [pts[i] if pts[i][1] > pts[i+1][1] else pts[i+1] for i in range(0, len(pts), 2)]
    floor_xy = _np_coor2xy(np.array(pts), z=50, coorW=W, coorH=H, floorW=W, floorH=W)
    floor_xy[:, 0] -= (W / 2 - 0.5)
    floor_xy[:, 1] -= (W / 2 - 0.5)
    floor_xy[:, 1] = -floor_xy[:, 1]
    return rectify_polygon(floor_xy).astype(np.float64)

class Wall:
    def __init__(self, val):
        self.val = val
        self.pts = []

def snap_walls_1d(walls, uf, threshold):
    srt = sorted(range(len(walls)), key=lambda idx: walls[idx].val)
    if not srt: return
    curr = [srt[0]]
    for idx in srt[1:]:
        if abs(walls[idx].val - walls[curr[0]].val) <= threshold:
            curr.append(idx)
        else:
            for c_idx in curr[1:]: uf.union(curr[0], c_idx)
            curr = [idx]
    for c_idx in curr[1:]: uf.union(curr[0], c_idx)

def apply_uf_means(walls, uf, polys_world, axis):
    groups = {}
    for i in range(len(walls)): groups.setdefault(uf.find(i), []).append(i)
    for root, indices in groups.items():
        mean_val = np.mean([walls[idx].val for idx in indices])
        for idx in indices:
            for poly_idx, p_idx in walls[idx].pts:
                polys_world[poly_idx][p_idx][axis] = mean_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wall_threshold", type=float, default=0.2)
    ap.add_argument("--rot90", action="store_true", help="Rotate visually 90 degrees clockwise")
    args = ap.parse_args()

    scene_dir, poses_path, matches_path = Path(args.scene_dir), Path(args.poses), Path(args.matches)
    label_map = get_room_labels(scene_dir)
    poses = json.loads(poses_path.read_text()).get("poses", {})
    matches = json.loads(matches_path.read_text())

    rooms = {}
    nodes = json.loads((scene_dir / "manifest.json").read_text()).get("nodes", [])
    for n in nodes:
        pid = n.get("pano_id", "")
        if pid in poses and (scene_dir / f"layout_gt/{pid}.txt").exists():
            poly_local = load_layout(scene_dir / f"layout_gt/{pid}.txt", 1024, 512)
            # Center and strip rotation locally
            x, y, th = float(poses[pid]["x"]), float(poses[pid]["y"]), float(poses[pid]["theta"])
            poly = se2_apply((x, y, th), poly_local)
            centroid = np.mean(poly, axis=0)
            poly_aligned = rectify_polygon(poly - centroid, rotate_back=False) + centroid
            rooms[pid] = poly_aligned

    if not rooms: return

    # BFS Translation Alignment
    adj = {pid: [] for pid in rooms}
    edges_info = {}
    for m in matches:
        s, d = m["src"].replace(".txt", ""), m["dst"].replace(".txt", "")
        if s in rooms and d in rooms:
            adj[s].append(d)
            adj[d].append(s)
            edges_info[(s, d)] = (m["idx_src"], m["idx_dst"])
            edges_info[(d, s)] = (m["idx_dst"], m["idx_src"])

    root = list(rooms.keys())[0]
    queue = [root]
    visited = {root}
    
    while queue:
        curr = queue.pop(0)
        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                # Translate nxt to align with curr using explicit matched points
                src_idxs, dst_idxs = edges_info[(curr, nxt)]
                pts_curr = np.array([rooms[curr][i-1] for i in src_idxs])
                pts_nxt = np.array([rooms[nxt][i-1] for i in dst_idxs])
                
                # Compute centroid of the matched points
                c_curr = np.mean(pts_curr, axis=0)
                c_nxt = np.mean(pts_nxt, axis=0)
                shift = c_curr - c_nxt
                
                # Shift the entire nxt room
                rooms[nxt] += shift
                # Note: any rooms attached downstream of nxt will just be shifted when they are visited!

    # Convert mapping back to list
    pids = list(rooms.keys())
    polys_world = [rooms[pid] for pid in pids]

    # Global Wall Snapping (1D)
    x_walls, y_walls = [], []
    for i, poly in enumerate(polys_world):
        N = len(poly)
        for j in range(N):
            p1, p2 = j, (j+1)%N
            if abs(poly[p1][0] - poly[p2][0]) < abs(poly[p1][1] - poly[p2][1]):
                w = Wall(poly[p1][0])
                w.pts.extend([(i, p1), (i, p2)])
                x_walls.append(w)
            else:
                w = Wall(poly[p1][1])
                w.pts.extend([(i, p1), (i, p2)])
                y_walls.append(w)

    if args.wall_threshold > 0:
        uf_x, uf_y = UnionFind(len(x_walls)), UnionFind(len(y_walls))
        snap_walls_1d(x_walls, uf_x, args.wall_threshold)
        snap_walls_1d(y_walls, uf_y, args.wall_threshold)
        apply_uf_means(x_walls, uf_x, polys_world, 0)
        apply_uf_means(y_walls, uf_y, polys_world, 1)

    if args.rot90:
        for i in range(len(polys_world)):
            rotated = np.zeros_like(polys_world[i])
            rotated[:, 0] = polys_world[i][:, 1]
            rotated[:, 1] = -polys_world[i][:, 0]
            polys_world[i] = rotated

    # Drawing
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    
    if HAS_SHAPELY and len(polys_world) > 1:
        union_poly = unary_union([Polygon(p).buffer(0) for p in polys_world])
        if union_poly.geom_type == 'MultiPolygon':
            for geom in union_poly.geoms:
                ax.plot(*geom.exterior.xy, color='black', linewidth=3)
        else:
            ax.plot(*union_poly.exterior.xy, color='black', linewidth=3)

    for i, poly in enumerate(polys_world):
        closed = np.vstack([poly, poly[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], color='black', linewidth=1.5, zorder=3)
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.3, zorder=2)
        cx, cy = np.mean(poly, axis=0) # pseudo-camera
        ax.text(cx, cy, get_display_label(pids[i], label_map), fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, pad=1), zorder=10)

    # 標示出 perfect_matches.json 所定義的對齊點，依據配對賦予獨立群組顏色
    cmap = plt.get_cmap("tab20")
    has_matches = False
    for m_idx, m in enumerate(matches):
        s, d = m["src"].replace(".txt", ""), m["dst"].replace(".txt", "")
        if s in pids and d in pids:
            xs_src, ys_src = [], []
            xs_dst, ys_dst = [], []
            s_idx = pids.index(s)
            d_idx = pids.index(d)
            for i in m["idx_src"]:
                xs_src.append(polys_world[s_idx][i-1][0])
                ys_src.append(polys_world[s_idx][i-1][1])
            for i in m["idx_dst"]:
                xs_dst.append(polys_world[d_idx][i-1][0])
                ys_dst.append(polys_world[d_idx][i-1][1])
            
            color = cmap(m_idx % 20)
            if xs_src:
                ax.scatter(xs_src, ys_src, marker='o', color=color, edgecolor='white', s=100, zorder=15)
            if xs_dst:
                ax.scatter(xs_dst, ys_dst, marker='^', color=color, edgecolor='white', s=100, zorder=15)
            
            if xs_src or xs_dst:
                has_matches = True
                
    if has_matches:
        ax.scatter([], [], marker='o', color='gray', edgecolor='white', s=100, label='Src Match')
        ax.scatter([], [], marker='^', color='gray', edgecolor='white', s=100, label='Dst Match')
        ax.legend(loc="upper right")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    plt.close(fig)
    print(f"[OK] Perfect Tree Blueprint drawn to {args.out}")

if __name__ == "__main__": main()
