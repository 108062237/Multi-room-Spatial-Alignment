#!/usr/bin/env python3
"""
Step 1: Convert manifest.json (edges_raw + hotspots_xy) into edges_measurements.json.

We create a relative SE(2) measurement z_ij = (dx, dy, dtheta) for each raw edge (i -> j).

MVP strategy:
- translation: centroid(hotspots_xy[j]) - centroid(hotspots_xy[i])
- rotation: dtheta = ang_j - ang_i + pi
- note: The geometric rotation constraint assumes cameras and the door are collinear.
  This is a heuristic approximation. We rely on robust optimization in Step 3
  (e.g., Huber loss) and layout priors (Step 6) to absorb this error.
- covariance: Dynamic based on physical distance (further = larger uncertanity).

Usage:
  python src/01_build_edges_from_hotspots.py \
    --manifest data/group/58472_Floor1/manifest.json \
    --out data/group/58472_Floor1/edges_measurements.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.geom import wrap_pi
from typing import Dict, Any, List, Tuple



def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def wrap_pi(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def compute_relative_transform(P_i: List[float], P_j: List[float]) -> Tuple[float, float, float]:
    """
    Given hotspot vectors P_i and P_j (from camera center to door),
    estimate relative transform (dx, dy, dtheta) from j to i.
    
    Heuristic Approximation:
    Assumes heading towards the door from i is opposite to heading from j.
    dtheta_ij = atan2(P_j) - atan2(P_i) + pi
    t_ij = P_i - R_ij * P_j
    """
    ang_i = math.atan2(P_i[1], P_i[0])
    ang_j = math.atan2(P_j[1], P_j[0])
    
    # 1. Rotation
    dtheta = wrap_pi(ang_j - ang_i + math.pi)
    
    # 2. Translation
    c = math.cos(dtheta)
    s = math.sin(dtheta)
    
    rx = c * P_j[0] - s * P_j[1]
    ry = s * P_j[0] + c * P_j[1]
    
    dx = P_i[0] - rx
    dy = P_i[1] - ry
    
    return dx, dy, dtheta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.json from Step 0")
    parser.add_argument("--out", type=str, required=True, help="Output edges_measurements.json path")
    parser.add_argument("--base_sigma_xy", type=float, default=0.8, help="Base sigma for translation (meters)")
    parser.add_argument("--base_sigma_theta_deg", type=float, default=20.0, help="Base sigma for rotation (degrees)")
    parser.add_argument(
        "--keep_only_existing_nodes",
        action="store_true",
        help="If set, drop edges where i or j is not found in nodes table",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)

    manifest = load_json(manifest_path)
    scene_id = manifest.get("scene_id", manifest_path.parent.name)

    nodes = manifest.get("nodes", [])
    node_map: Dict[str, Dict[str, Any]] = {n["pano_id"]: n for n in nodes}

    edges_raw = manifest.get("edges_raw", [])
    if len(edges_raw) == 0:
        raise RuntimeError("manifest.edges_raw is empty. Did Step 0 parse HOTSPOT.json correctly?")

    base_sigma_theta = math.radians(args.base_sigma_theta_deg)

    edges_out: List[Dict[str, Any]] = []
    dropped = 0

    for e in edges_raw:
        i = e["src"]
        j = e["dst"]

        if args.keep_only_existing_nodes and (i not in node_map or j not in node_map):
            dropped += 1
            continue

        ni = node_map.get(i, {})
        nj = node_map.get(j, {})

        # Use index matching to prevent grabbing the wrong door if multiple exist
        idx_i = e.get("meta", {}).get("hotspot_index")
        
        conn_i = next((c for c in ni.get("connections", []) if c["neighbor"] == j and (idx_i is None or c.get("index_in_json") == idx_i)), None)
        
        # We find the reciprocal connection in j that points to i using distance/geometry or assuming single matched door.
        # Simplest reciprocal check: find a connection to i. If multiple, assume symmetric index.
        # But our manifest.edges_raw records the 'src' index. For proper matching, manifest could be better,
        # but for now we grab the connection to i. If multiple, we just use the first one matching neighbor==i.
        conn_j = next((c for c in nj.get("connections", []) if c["neighbor"] == i), None)

        if conn_i is None or conn_j is None:
            dropped += 1
            continue

        Hi = conn_i["hotspot_xy"]  # [x, y]
        Hj = conn_j["hotspot_xy"]  # [x, y]

        dx, dy, dtheta = compute_relative_transform(Hi, Hj)
        
        # Dynamic Confidence based on physical distance to the door
        # Longer distance -> higher uncertainty (larger sigma)
        dist_i = math.hypot(Hi[0], Hi[1])
        dist_j = math.hypot(Hj[0], Hj[1])
        avg_dist = max(0.5, (dist_i + dist_j) / 2.0)  # Prevent zero div
        
        # Penalty multiplier scales linearly with distance (if > 1m)
        dist_penalty = max(1.0, avg_dist)

        sigma_xy = args.base_sigma_xy * dist_penalty
        sigma_theta = base_sigma_theta * dist_penalty

        # Covariance diagonal representation [var_x, var_y, var_theta]
        cov = [sigma_xy**2, sigma_xy**2, sigma_theta**2]

        edges_out.append(
            {
                "i": i,
                "j": j,
                "measurement": {"dx": float(dx), "dy": float(dy), "dtheta": float(dtheta)},
                "noise_sigma": {"sigma_xy": float(sigma_xy), "sigma_theta": float(sigma_theta)},
                "covariance_diag": cov,
                "meta": {
                    "source": "hotspot_geometric_heuristic",
                    "avg_hotspot_dist": float(avg_dist),
                    "hotspot_idx_i": conn_i.get("index_in_json"),
                    "hotspot_idx_j": conn_j.get("index_in_json"),
                },
            }
        )

    out = {
        "scene_id": scene_id,
        "units": {"translation": "meters", "rotation": "radians"},
        "params": {
            "base_sigma_xy": float(args.base_sigma_xy),
            "base_sigma_theta_deg": float(args.base_sigma_theta_deg),
            "note": "Geometric Edges (Heuristic): Uses distance-scaled covariance.",
        },
        "stats": {
            "num_nodes": len(node_map),
            "num_edges_raw": len(edges_raw),
            "num_edges_written": len(edges_out),
            "num_edges_dropped": int(dropped),
        },
        "edges": edges_out,
    }

    save_json(out_path, out)
    print(f"[OK] Wrote {len(edges_out)} edges -> {out_path}")
    print(f"Stats: {out['stats']}")


if __name__ == "__main__":
    main()
