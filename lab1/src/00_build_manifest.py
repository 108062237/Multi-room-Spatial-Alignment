#!/usr/bin/env python3
"""
Step 0: Build manifest.json for each scene folder under data/group/*

Key policies:
- Node universe is ONLY panos/ images (ignores case for extensions).
- Stores paths relative to scene_dir.
- Ensures robust JSON parsing for single-item dicts.
- Enforces bidirectional edges (Graph must be symmetric).
- Identifies and optionally flags Orphan nodes.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def find_hotspot_json(scene_dir: Path) -> Path:
    cands = sorted(scene_dir.glob("*HOTSPOT.json"))
    if not cands:
        raise FileNotFoundError(f"No *HOTSPOT.json found under {scene_dir}")
    return cands[0]


def find_layout_gt_file(layout_gt_dir: Path, pano_id: str) -> Optional[Path]:
    """
    Try to find GT layout file for pano_id in layout_gt/.
    Try common extensions; if not found, fallback to regex search.
    """
    exts = [".json", ".txt", ".npy", ".npz"]
    for ext in exts:
        p = layout_gt_dir / f"{pano_id}{ext}"
        if p.exists():
            return p

    # Fallback using regex to ensure exact pano_id match (avoids substring issues)
    pattern = re.compile(rf"(^|[^a-zA-Z0-9]){pano_id}([^a-zA-Z0-9]|$)")
    matches = []
    for f in layout_gt_dir.iterdir():
        if f.is_file() and pattern.search(f.name):
            matches.append(f)
            
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # prefer exact stem match
        for m in matches:
            if m.stem == pano_id:
                return m
        return sorted(matches)[0]
    return None


def build_manifest_for_scene(scene_dir: Path) -> Path:
    panos_dir = scene_dir / "panos"
    layout_gt_dir = scene_dir / "layout_gt"
    if not panos_dir.exists():
        raise FileNotFoundError(f"Missing panos/ in {scene_dir}")
    if not layout_gt_dir.exists():
        raise FileNotFoundError(f"Missing layout_gt/ in {scene_dir}")

    hotspot_json = find_hotspot_json(scene_dir)
    hotspot_data = json.loads(hotspot_json.read_text())
    hotspot_items = hotspot_data.get("HOTSPOTOFROOM", [])

    # 1) Node universe = images in panos/ (case-insensitive extension match)
    valid_exts = {'.jpg', '.jpeg', '.png'}
    img_paths = sorted([p for p in panos_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])
    if not img_paths:
        raise FileNotFoundError(f"No pano images found under {panos_dir}")

    nodes: Dict[str, Dict[str, Any]] = {}
    for img_path in img_paths:
        pano_id = img_path.stem
        layout_path = find_layout_gt_file(layout_gt_dir, pano_id)
        
        # Save RELATIVE paths to avoid hardcoded absolute path issues
        rel_img = img_path.relative_to(scene_dir)
        rel_layout = layout_path.relative_to(scene_dir) if layout_path else None
        
        nodes[pano_id] = {
            "pano_id": pano_id,
            "image_path": str(rel_img),
            "layout_gt_path": str(rel_layout) if rel_layout else None,
            "room_idx": None,
            "is_orphan": True,  # Will turn False if connected
            "connections": [], 
        }

    pano_universe = set(nodes.keys())

    # 2) First pass: Collect all raw connections
    raw_connections_map = {}
    ignored_items = 0
    ignored_neighbors = 0

    for item in hotspot_items:
        pano_id = Path(item.get("IDName", "")).stem
        if not pano_id or pano_id not in pano_universe:
            ignored_items += 1
            continue

        nodes[pano_id]["room_idx"] = int(item.get("Roomidx", -1))
        
        # Robust parsing: ensure list even if single item parsed as dict/string
        coords_3d = item.get("HSLoc_3D", {}).get("Coordinate_3D", [])
        neigh_list = item.get("ToIDName", {}).get("IDName", [])
        
        if isinstance(coords_3d, dict):
            coords_3d = [coords_3d]
        if isinstance(neigh_list, str):
            neigh_list = [neigh_list]
            
        count = min(len(coords_3d), len(neigh_list))

        for k in range(count):
            p3 = coords_3d[k]
            dst_id = Path(neigh_list[k]).stem

            if dst_id not in pano_universe:
                ignored_neighbors += 1
                continue
            
            hx, hy = float(p3["x"]), float(p3["y"])
            if pano_id not in raw_connections_map:
                raw_connections_map[pano_id] = {}
            raw_connections_map[pano_id][dst_id] = (hx, hy, k)

    # 3) Second pass: Enforce Bidirectional Edges & Orphans
    edges_raw: List[Dict[str, Any]] = []
    
    for src_id, connections in raw_connections_map.items():
        for dst_id, (hx, hy, k) in connections.items():
            # Check if dst -> src exists (Bidirectional requirement)
            if dst_id in raw_connections_map and src_id in raw_connections_map[dst_id]:
                # It is a valid bidirectional edge
                nodes[src_id]["is_orphan"] = False
                nodes[dst_id]["is_orphan"] = False
                
                nodes[src_id]["connections"].append({
                    "neighbor": dst_id,
                    "hotspot_xy": [hx, hy],
                    "index_in_json": k
                })
                
                edges_raw.append({
                    "src": src_id,
                    "dst": dst_id,
                    "src_room_idx": nodes[src_id]["room_idx"],
                    "via": "hotspot_link",
                    "meta": {"hotspot_index": k}
                })

    # Stats
    num_nodes = len(nodes)
    num_edges = len(edges_raw)
    num_layout = sum(1 for n in nodes.values() if n["layout_gt_path"] is not None)
    num_connections = sum(len(n["connections"]) for n in nodes.values())
    num_orphans = sum(1 for n in nodes.values() if n["is_orphan"])
    
    # Graph connectedness check
    is_fully_connected = (num_nodes > 0 and num_orphans == 0)

    manifest = {
        "scene_id": scene_dir.name,
        "scene_dir": str(scene_dir.absolute()),
        "panos_dir_rel": "panos",
        "layout_gt_dir_rel": "layout_gt",
        "hotspot_json_rel": str(hotspot_json.name),
        "nodes": list(nodes.values()),
        "edges_raw": edges_raw,
        "stats": {
            "num_nodes": num_nodes,
            "num_edges_raw": num_edges,
            "num_nodes_with_layout_gt": num_layout,
            "num_orphan_nodes": num_orphans,
            "is_fully_connected": is_fully_connected,
            "total_valid_connections_bidirectional": num_connections,
            "ignored_hotspot_items_not_in_panos": ignored_items,
            "ignored_neighbors_not_in_panos": ignored_neighbors,
        },
    }

    out_path = scene_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_root", type=str, required=True, help="Path to data/group")
    parser.add_argument("--scene", type=str, default="", help="If set, only process this scene folder name.")
    args = parser.parse_args()

    group_root = Path(args.group_root)
    if not group_root.exists():
        raise FileNotFoundError(group_root)

    if args.scene:
        scene_dirs = [group_root / args.scene]
        if not scene_dirs[0].exists():
            raise FileNotFoundError(scene_dirs[0])
    else:
        scene_dirs = sorted([p for p in group_root.iterdir() if p.is_dir()])

    ok, fail = 0, 0
    for scene_dir in scene_dirs:
        try:
            out_path = build_manifest_for_scene(scene_dir)
            
            m = json.loads(Path(out_path).read_text())
            conn_status = "FULLY CONNECTED" if m["stats"]["is_fully_connected"] else f"HAS {m['stats']['num_orphan_nodes']} ORPHANS"
            print(f"[OK] {scene_dir.name} -> {out_path.name} | {conn_status}")
            print("     stats:", m["stats"])
            ok += 1
        except Exception as e:
            print(f"[FAIL] {scene_dir.name}: {e}")
            fail += 1

    print(f"Done. OK={ok}, FAIL={fail}")


if __name__ == "__main__":
    main()
