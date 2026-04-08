#!/usr/bin/env python3
"""
Step 4: Visualize pose graph before/after optimization.

Always works:
- draw nodes (poses) and edges (measurements) in XY plane
Optionally:
- attempt to draw layout polygons if layout_gt format is recognized (best-effort)

Usage:
  python src/04_viz_pose_graph.py \
    --scene_dir data/group/58472_Floor1 \
    --before   data/group/58472_Floor1/initial_poses.json \
    --after    data/group/58472_Floor1/optimized_poses.json \
    --edges    data/group/58472_Floor1/edges_measurements.json \
    --out_dir  data/group/58472_Floor1/viz
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import matplotlib
matplotlib.use("Agg")

# For Chinese font support fallback
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.geom import se2_apply
from src.utils.labels import get_room_labels, get_display_label


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def se2_apply(pose: Tuple[float, float, float], pts: np.ndarray) -> np.ndarray:
    """Apply SE2 pose (x,y,theta) to Nx2 points."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([x, y], dtype=np.float64)


# -----------------------------
# Optional: layout parsing (best-effort)
# -----------------------------
def _pano_xy_to_u_v(x: float, y: float, W: int, H: int):
    u = ((x + 0.5) / W - 0.5) * 2.0 * math.pi
    v = -((y + 0.5) / H - 0.5) * math.pi
    return u, v


def _ray_from_uv(u: float, v: float) -> np.ndarray:
    cu, su = math.cos(u), math.sin(u)
    cv, sv = math.cos(v), math.sin(v)
    return np.array([cv * cu, cv * su, sv], dtype=np.float64)


def _intersect_with_z_plane(dir3: np.ndarray, z_plane: float = -1.0) -> Optional[np.ndarray]:
    dz = dir3[2]
    if abs(dz) < 1e-8:
        return None
    t = z_plane / dz
    # We only accept intersections in front of camera pointing DOWN to the floor (t>0)
    if t <= 0:
        return None
    return t * dir3


def try_load_layout_polygon(layout_path: Path) -> Optional[np.ndarray]:
    if layout_path is None or not layout_path.exists():
        return None

    suf = layout_path.suffix.lower()

    try:
        if suf == ".txt":
            W, H = 1024, 512
            z_floor = -1.0 

            lines = []
            for line in layout_path.read_text().splitlines():
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
                # In images, y=0 is top (ceiling), y=H is bottom (floor).
                # So we pick the pixel with the MAX y-coordinate.
                if p1[1] > p2[1]:
                    floor_pixels.append(p1)
                else:
                    floor_pixels.append(p2)

            pts_xy = []
            for (px, py) in floor_pixels:
                u, v = _pano_xy_to_u_v(px, py, W, H)
                dir3 = _ray_from_uv(u, v)
                p3 = _intersect_with_z_plane(dir3, z_plane=z_floor)
                if p3 is None:
                    continue
                pts_xy.append([p3[0], p3[1]])

            if len(pts_xy) < 3:
                return None

            return np.array(pts_xy, dtype=np.float64)

        if suf == ".json":
            d = json.loads(layout_path.read_text())
            for k in ["polygon", "points", "uv", "corners", "xy"]:
                if k in d and isinstance(d[k], list) and len(d[k]) >= 3:
                    arr = np.array(d[k], dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        return arr[:, :2]
        if suf == ".npy":
            arr = np.load(str(layout_path))
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 3:
                return arr[:, :2].astype(np.float64)

    except Exception:
        return None

    return None



# -----------------------------
# Plotting
# -----------------------------
def plot_graph(
    ax,
    poses: Dict[str, Dict[str, float]],
    edges: List[Dict[str, Any]],
    title: str,
    draw_edges: bool = True,
    draw_labels: bool = True,
    manifest_nodes: Optional[List[Dict[str, Any]]] = None,
    label_map: Optional[Dict[str, str]] = None
):
    ids = sorted(poses.keys())
    xy = np.array([[poses[i]["x"], poses[i]["y"]] for i in ids], dtype=np.float64)
    ax.scatter(xy[:, 0], xy[:, 1], s=40, zorder=5)

    # yaw arrow
    for i in ids:
        x, y, th = poses[i]["x"], poses[i]["y"], poses[i]["theta"]
        ax.arrow(x, y, 0.25 * math.cos(th), 0.25 * math.sin(th), 
                 head_width=0.06, length_includes_head=True, zorder=6, color='black')

    # Draw smart edges (Camera i -> Door -> Camera j) if connection data exists
    if draw_edges:
        # Build node quick lookup for connections if manifest_nodes provided
        node_map = {n['pano_id']: n for n in manifest_nodes} if manifest_nodes else {}
        
        for e in edges:
            i, j = e["i"], e["j"]
            if i not in poses or j not in poses:
                continue
                
            xi, yi, thi = poses[i]["x"], poses[i]["y"], poses[i]["theta"]
            xj, yj, thj = poses[j]["x"], poses[j]["y"], poses[j]["theta"]
            
            # Identify hotspot to draw dog-leg line
            hotspot_coords = None
            if node_map and i in node_map:
                conn_i = next((c for c in node_map[i].get("connections", []) if c["neighbor"] == j), None)
                if conn_i:
                    hx, hy = conn_i["hotspot_xy"]
                    # Transform local hotspot to global
                    c_i, s_i = math.cos(thi), math.sin(thi)
                    gx = xi + c_i * hx - s_i * hy
                    gy = yi + s_i * hx + c_i * hy
                    hotspot_coords = (gx, gy)
            
            if hotspot_coords:
                gx, gy = hotspot_coords
                # Plot Camera i -> Door
                ax.plot([xi, gx], [yi, gy], color='blue', linewidth=1, alpha=0.6, zorder=3)
                # Plot Door -> Camera j
                ax.plot([gx, xj], [gy, yj], color='blue', linewidth=1, alpha=0.6, zorder=3)
                # Mark Door
                ax.scatter([gx], [gy], marker='s', color='#FFA500', s=15, zorder=4)
            else:
                # Fallback to direct Camera -> Camera edge
                ax.plot([xi, xj], [yi, yj], color='blue', linewidth=1, alpha=0.6, zorder=3)

    if draw_labels:
        for i in ids:
            label_text = get_display_label(i, label_map) if label_map else i[-6:]
            ax.text(poses[i]["x"] + 0.1, poses[i]["y"] + 0.1, label_text, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                    zorder=10)

    ax.set_title(title)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_layouts_if_possible(
    ax,
    scene_dir: Path,
    manifest_nodes: List[Dict[str, Any]],
    poses: Dict[str, Dict[str, float]],
    alpha: float = 0.25,
):
    drawn = 0
    for n in manifest_nodes:
        pid = n["pano_id"]
        if pid not in poses:
            continue
        lp = n.get("layout_gt_path")
        if not lp:
            continue
            
        lp_path = Path(lp)
        if not lp_path.is_absolute():
            lp_path = scene_dir / lp_path
            
        poly_local = try_load_layout_polygon(lp_path)
        if poly_local is None:
            continue

        pose = (poses[pid]["x"], poses[pid]["y"], poses[pid]["theta"])
        poly_world = se2_apply(pose, poly_local)

        poly_world = np.vstack([poly_world, poly_world[0:1]])
        ax.plot(poly_world[:, 0], poly_world[:, 1], color='black', linewidth=1.5, alpha=alpha, zorder=2)
        drawn += 1

    return drawn


def plot_error_quiver(ax, before_poses: Dict[str, Dict[str, float]], after_poses: Dict[str, Dict[str, float]]):
    """Draw arrows pointing from Before pose to After pose."""
    for i in before_poses.keys():
        if i in after_poses:
            bx, by = before_poses[i]["x"], before_poses[i]["y"]
            afx, afy = after_poses[i]["x"], after_poses[i]["y"]
            # Draw gray dashed arrow indicating movement
            ax.annotate("", xy=(afx, afy), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="->", color="gray", linewidth=1, alpha=0.7, linestyle="dashed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", type=str, required=True, help="data/group/<scene>")
    ap.add_argument("--before", type=str, required=True, help="initial_poses.json")
    ap.add_argument("--after", type=str, required=True, help="optimized_poses.json")
    ap.add_argument("--edges", type=str, required=True, help="edges_measurements.json")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory for images")
    ap.add_argument("--no_labels", action="store_true", help="hide node labels")
    ap.add_argument("--no_edges", action="store_true", help="hide edges")
    ap.add_argument("--draw_layouts", action="store_true", help="try to draw layout_gt polygons (best-effort)")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    manifest = load_json(scene_dir / "manifest.json")
    nodes = manifest.get("nodes", [])

    before = load_json(Path(args.before))
    after = load_json(Path(args.after))
    edges_data = load_json(Path(args.edges))
    edges = edges_data.get("edges", [])

    poses_before = before.get("poses", {})
    poses_after = after.get("poses", {})
    
    label_map = get_room_labels(scene_dir)

    # 1) before only
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_graph(
        ax,
        poses_before,
        edges,
        title=f"{scene_dir.name} - BEFORE (initial)",
        draw_edges=not args.no_edges,
        draw_labels=not args.no_labels,
        manifest_nodes=nodes,
        label_map=label_map
    )
    drawn = 0
    if args.draw_layouts:
        drawn = plot_layouts_if_possible(ax, scene_dir, nodes, poses_before, alpha=0.35)
        ax.set_title(ax.get_title() + f" | layouts drawn: {drawn}")
        print(f"[INFO] layouts drawn (before): {drawn}")
    fig.tight_layout()
    fig.savefig(out_dir / "before.png", dpi=200)
    plt.close(fig)

    # 2) after only
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plot_graph(
        ax,
        poses_after,
        edges,
        title=f"{scene_dir.name} - AFTER (optimized)",
        draw_edges=not args.no_edges,
        draw_labels=not args.no_labels,
        manifest_nodes=nodes,
        label_map=label_map
    )
    drawn = 0
    if args.draw_layouts:
        drawn = plot_layouts_if_possible(ax, scene_dir, nodes, poses_after, alpha=0.35)
        ax.set_title(ax.get_title() + f" | layouts drawn: {drawn}")
        print(f"[INFO] layouts drawn (after): {drawn}")
    fig.tight_layout()
    fig.savefig(out_dir / "after.png", dpi=200)
    plt.close(fig)

    # 3) overlay
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # background (before) - muted
    b_edges = not args.no_edges
    b_labels = not args.no_labels
    
    # Plot Quiver before underlying graph to keep it clean
    plot_error_quiver(ax, poses_before, poses_after)

    plot_graph(
        ax, poses_before, edges, title="", draw_edges=b_edges, draw_labels=False, manifest_nodes=nodes, label_map=label_map
    )
    # Dim the Before graph manually
    for collection in ax.collections:
        collection.set_alpha(0.15)
    for line in ax.lines:
        line.set_alpha(0.15)

    if args.draw_layouts:
        plot_layouts_if_possible(ax, scene_dir, nodes, poses_before, alpha=0.1)

    # foreground (after)
    plot_graph(
        ax, poses_after, edges, title=f"{scene_dir.name} - OVERLAY", draw_edges=b_edges, draw_labels=b_labels, manifest_nodes=nodes, label_map=label_map
    )
    if args.draw_layouts:
        plot_layouts_if_possible(ax, scene_dir, nodes, poses_after, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "overlay.png", dpi=200)
    plt.close(fig)

    print(f"[OK] Wrote viz images -> {out_dir}")


if __name__ == "__main__":
    main()
