import argparse
import matplotlib
matplotlib.use('Agg')
import json
import sys
import numpy as np
import math
from pathlib import Path

# 載入 LayoutHub 工具
_LAYOUTHUB = Path(__file__).resolve().parent.parent.parent / 'LayoutHub'
if _LAYOUTHUB.exists():
    sys.path.insert(0, str(_LAYOUTHUB))

from utils.post_proc import np_coor2xy

def load_layout_txt(txt_path, pano_w=1024, pano_h=512, z=50):
    """與 tool_pairwise_verifier.py 完全一致的投影方法"""
    path = Path(txt_path)
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    if len(pts) < 2:
        raise ValueError(f"TXT 點數不足: {txt_path}")

    if len(pts) % 2 == 0:
        floor_pixel = [pts[i] if pts[i][1] > pts[i+1][1] else pts[i+1]
                       for i in range(0, len(pts), 2)]
    else:
        floor_pixel = pts

    floor_pixel = np.array(floor_pixel)
    floor_xy = np_coor2xy(floor_pixel, z=z,
                          coorW=pano_w, coorH=pano_h,
                          floorW=pano_w, floorH=pano_w)
    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]   # Y 翻轉
    return floor_xy

def compute_relative_pose(pA_start, pA_end, pB_start, pB_end):
    """計算 B 對齊到 A 的 dx, dy, dtheta（不加 + pi，與 verifier 一致）"""
    angA = math.atan2(pA_end[1] - pA_start[1], pA_end[0] - pA_start[0])
    angB = math.atan2(pB_end[1] - pB_start[1], pB_end[0] - pB_start[0])
    dtheta = angA - angB
    dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi  # 限制在 [-π, π]

    c, s = math.cos(dtheta), math.sin(dtheta)
    rx = c * pB_start[0] - s * pB_start[1]
    ry = s * pB_start[0] + c * pB_start[1]
    dx = pA_start[0] - rx
    dy = pA_start[1] - ry
    return dx, dy, dtheta

def main():
    parser = argparse.ArgumentParser(description="從完美的配對 JSON 產生 GTSAM edges.json")
    parser.add_argument('--matches', required=True, help='輸入的配對檔案 (例: perfect_matches.json)')
    parser.add_argument('--layout_dir', required=True, help='房間 txt 佈局檔所在的資料夾路徑 (例: layout_gt/)')
    parser.add_argument('--out', required=True, help='輸出的 edges JSON 檔案 (例: perfect_gtsam_edges.json)')
    parser.add_argument('--scene_id', required=False, help='場景 ID (不填則從 layout_dir 自動推斷)')
    args = parser.parse_args()

    matches_json_path = args.matches
    base_dir = Path(args.layout_dir)
    out_edges_path = args.out
    scene_id = args.scene_id if args.scene_id else base_dir.parent.name

    matches = json.load(open(matches_json_path, 'r'))
    edges = []

    for edge in matches:
        src_file, dst_file = edge["src"], edge["dst"]
        # 轉成無副檔名 ID（GTSAM 腳本用的格式）
        src_id = src_file.replace(".txt", "")
        dst_id = dst_file.replace(".txt", "")

        poly_src = load_layout_txt(base_dir / src_file)
        poly_dst = load_layout_txt(base_dir / dst_file)

        pA_start = poly_src[edge["idx_src"][0] - 1]
        pA_end   = poly_src[edge["idx_src"][1] - 1]
        pB_start = poly_dst[edge["idx_dst"][0] - 1]
        pB_end   = poly_dst[edge["idx_dst"][1] - 1]

        dx, dy, dtheta = compute_relative_pose(pA_start, pA_end, pB_start, pB_end)

        # 完美匹配：使用非常小的雜訊（高置信度）
        edges.append({
            "i": src_id,
            "j": dst_id,
            "measurement": {
                "dx": dx,
                "dy": dy,
                "dtheta": dtheta
            },
            "noise_sigma": {
                "sigma_xy": 0.05,           # 很小 = 高置信度
                "sigma_theta": 0.017453     # ~1 度
            },
            "meta": {
                "source": "perfect_manual_match",
                "confidence": 1.0
            }
        })

    output = {
        "scene_id": scene_id,
        "edges": edges
    }

    with open(out_edges_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"✅ 成功產生 {len(edges)} 條完美邊界: {out_edges_path}")

if __name__ == "__main__":
    main()