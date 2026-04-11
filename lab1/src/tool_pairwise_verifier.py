import argparse
import json
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add LayoutHub directory so we can import its utils
_LAYOUTHUB = Path(__file__).parent.parent.parent / 'LayoutHub'
if _LAYOUTHUB.exists():
    sys.path.insert(0, str(_LAYOUTHUB))

try:
    from utils.geom import rectify_polygon, align_to_manhattan
except ImportError:
    pass  # We can still run without it if not found there

try:
    from utils.post_proc import np_coor2xy
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    print("[警告] 無法載入 utils.post_proc（LayoutHub 路徑找不到）。")

# ==========================================
# 1. 檔案讀取器 (Loaders)
# ==========================================

def load_layout_txt(txt_path, pano_w=1024, pano_h=512, z=50):
    """
    讀取 TXT 格式，並用 np_coor2xy 投影到真正的 3D 地板 XY 座標。
    TXT 格式：每對兩行 = (天花板像素座標, 地板像素座標)，在 1024x512 全景座標系中。
    直接用像素座標畫圖是錯誤的（Y 值全都在 320-360 之間，導致一條橫線）。
    必須用 np_coor2xy 將全景角度投影到俯視 2D 平面。
    """
    if not HAS_UTILS:
        raise ImportError("缺少 utils.post_proc（LayoutHub 路徑錯誤），無法投影 TXT 座標！")

    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {txt_path}")

    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])

    if len(pts) < 2:
        raise ValueError(f"TXT 點數不足: {txt_path}")

    # 交錯格式 (天花板, 地板, 天花板, 地板...)
    if len(pts) % 2 == 0:
        # 取地板點：y 值較大的那個（像素 y 越大越接近圖片底部 = 地板）
        floor_pixel = []
        for i in range(0, len(pts), 2):
            p1, p2 = pts[i], pts[i + 1]
            floor_pixel.append(p1 if p1[1] > p2[1] else p2)
    else:
        floor_pixel = pts

    floor_pixel = np.array(floor_pixel)

    # 用 np_coor2xy 投影到真正的 3D 地板 XY 座標
    # 這等同於 visualize_combined_corners.py 的做法
    floor_xy = np_coor2xy(
        floor_pixel, z=z,
        coorW=pano_w, coorH=pano_h,
        floorW=pano_w, floorH=pano_w
    )

    # 移動原點到相機位置 (0, 0)
    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center

    # 翻轉 Y 使順時針排列正確（與 visualize_combined_corners.py 一致）
    floor_xy[:, 1] = -floor_xy[:, 1]

    return floor_xy


def get_floor_xy_from_json(json_path, W=1024, H=512):
    """從 JSON 的 UV 座標投影出地板 2D 座標"""
    if not HAS_UTILS:
        raise ImportError("缺少 utils.post_proc，無法解析 JSON 格式！")

    data = json.load(open(json_path, 'r'))
    if 'uv' not in data:
        raise ValueError(f"JSON 缺少 'uv' 鍵值: {json_path}")

    uv = np.array(data['uv'])
    z1 = data.get('z1', 50)

    uv_img = uv.copy()
    uv_img[:, 0] *= W
    uv_img[:, 1] *= H

    floor_indices = range(1, len(uv), 2)
    floor_uv_img = uv_img[floor_indices]

    floor_xy = np_coor2xy(floor_uv_img, z=z1, coorW=W, coorH=H, floorW=W, floorH=W)
    center = W / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]  # Y 軸反轉
    
    try:
        floor_xy = rectify_polygon(floor_xy)
    except Exception:
        pass
        
    return floor_xy


def load_room_geometry(file_path, W=1024, H=512):
    """通用路由：自動判斷副檔名並呼叫對應的 Loader"""
    path = Path(file_path)
    if path.suffix.lower() == '.json':
        print(f"  [JSON] 執行 3D 投影解析: {path.name}")
        return get_floor_xy_from_json(path, W, H)
    elif path.suffix.lower() == '.txt':
        print(f"  [TXT]  投影全景像素座標 → 3D 地板 XY: {path.name}")
        return load_layout_txt(path, pano_w=1024, pano_h=512)
    else:
        raise ValueError(f"不支援的格式: {path.suffix}")


# ==========================================
# 2. 幾何運算核心 (Math & Geometry)
# ==========================================
# align_to_manhattan has been moved to utils/geom.py

def main():
    parser = argparse.ArgumentParser(description="通用版雙房間精準拼接 (支援 TXT/JSON)")
    parser.add_argument('--room_a', required=True, help='基準房間 (Room A) 的檔案路徑 (.txt 或 .json)')
    parser.add_argument('--room_b', required=True, help='要拼上去的房間 (Room B) 的檔案路徑 (.txt 或 .json)')
    parser.add_argument('--idx_a', type=int, nargs=2, required=True, help='Room A 牆壁圖上標示的數字 (例: 1 4)')
    parser.add_argument('--idx_b', type=int, nargs=2, required=True, help='Room B 牆壁圖上標示的數字 (例: 3 6)')
    parser.add_argument('--out', default='verify_universal.png', help='輸出圖片檔名')
    parser.add_argument('--invert_y', action='store_true', help='加上此參數可反轉 Y 軸 (讓圖形上下顛倒)')
    parser.add_argument('--W', type=int, default=1024, help='全景圖寬度 (僅 JSON 適用)')
    parser.add_argument('--H', type=int, default=512, help='全景圖高度 (僅 JSON 適用)')
    args = parser.parse_args()

    # 1. 萬用讀取器：根據副檔名自動處理
    poly_A_raw = load_room_geometry(args.room_a, args.W, args.H)
    poly_B = load_room_geometry(args.room_b, args.W, args.H)

    # 將基準房間轉正，並取得旋轉矩陣（用於同步旋轉相機位置）
    poly_A, R_A = align_to_manhattan(poly_A_raw)

    # 相機 A 在 Room A 自身座標系是 (0,0)，套用同樣旋轉
    cam_A_aligned = np.array([0.0, 0.0]) @ R_A.T  # 仍是 (0,0)，但方向正確

    # 2. 自動處理 1-based 到 0-based 的 Index 轉換
    pA_start, pA_end = poly_A[args.idx_a[0] - 1], poly_A[args.idx_a[1] - 1]
    pB_start, pB_end = poly_B[args.idx_b[0] - 1], poly_B[args.idx_b[1] - 1]

    # 3. 計算 SE(2) 對齊矩陣
    # 移除 + math.pi，讓 B_start→B_end 與 A_start→A_end 同向
    # 結果：A_start↔B_start，A_end↔B_end（即 A1↔B2，A4↔B3）
    angA = math.atan2(pA_end[1] - pA_start[1], pA_end[0] - pA_start[0])
    angB = math.atan2(pB_end[1] - pB_start[1], pB_end[0] - pB_start[0])
    dtheta = angA - angB

    c, s = math.cos(dtheta), math.sin(dtheta)
    rx, ry = c * pB_start[0] - s * pB_start[1], s * pB_start[0] + c * pB_start[1]
    dx, dy = pA_start[0] - rx, pA_start[1] - ry

    # 4. 變換 Room B 座標
    poly_B_aligned = (poly_B @ np.array([[c, -s], [s, c]]).T) + np.array([dx, dy])

    # 相機 B 原點 (0,0) 套用同樣的 SE(2) 變換
    cam_B_aligned = (np.array([0.0, 0.0]) @ np.array([[c, -s], [s, c]]).T) + np.array([dx, dy])
    real_cam_dist = math.hypot(cam_B_aligned[0] - cam_A_aligned[0], cam_B_aligned[1] - cam_A_aligned[1])

    # 5. 畫圖驗證
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Universal Alignment | Camera Dist: {real_cam_dist:.2f} px", fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    if args.invert_y:
        ax.invert_yaxis()

    # 畫 Room A（藍色）+ 角點數字（標籤朝 Room A 質心方向偏移）
    pA_c = np.vstack([poly_A, poly_A[0]])
    ax.plot(pA_c[:, 0], pA_c[:, 1], 'b-', linewidth=3, label='Room A (Base)')
    ax.fill(poly_A[:, 0], poly_A[:, 1], 'blue', alpha=0.1)
    cA = poly_A.mean(axis=0)  # Room A 質心
    label_offset = max(np.ptp(poly_A, axis=0)) * 0.06
    for i, (x, y) in enumerate(poly_A):
        ax.plot(x, y, 'bo', markersize=8)
        dx = cA[0] - x; dy = cA[1] - y
        norm = math.hypot(dx, dy) or 1
        ox, oy = dx / norm * label_offset, dy / norm * label_offset
        ax.text(x + ox, y + oy, f'A{i+1}', fontsize=11, fontweight='bold',
                color='blue', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # 畫 Room B（紅色）+ 角點數字（標籤朝 Room B 質心方向偏移）
    pB_c = np.vstack([poly_B_aligned, poly_B_aligned[0]])
    ax.plot(pB_c[:, 0], pB_c[:, 1], 'r-', linewidth=3, label='Room B (Aligned)')
    ax.fill(poly_B_aligned[:, 0], poly_B_aligned[:, 1], 'red', alpha=0.1)
    cB = poly_B_aligned.mean(axis=0)  # Room B 質心
    label_offset_b = max(np.ptp(poly_B_aligned, axis=0)) * 0.06
    for i, (x, y) in enumerate(poly_B_aligned):
        ax.plot(x, y, 'rs', markersize=8)
        dx = cB[0] - x; dy = cB[1] - y
        norm = math.hypot(dx, dy) or 1
        ox, oy = dx / norm * label_offset_b, dy / norm * label_offset_b
        ax.text(x + ox, y + oy, f'B{i+1}', fontsize=11, fontweight='bold',
                color='red', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # 相機與共用牆起點
    ax.scatter(cam_A_aligned[0], cam_A_aligned[1], marker='^', color='blue', s=200, zorder=5, label='Camera A')
    ax.scatter(cam_B_aligned[0], cam_B_aligned[1], marker='^', color='red',  s=200, zorder=5, label='Camera B')
    ax.plot([cam_A_aligned[0], cam_B_aligned[0]], [cam_A_aligned[1], cam_B_aligned[1]], 'k--', alpha=0.5)
    ax.plot(pA_start[0], pA_start[1], 'g*', markersize=16, label='Matched Corner')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight')
    print(f"\n✅ 拼接完成！圖片已儲存至: {args.out}")

if __name__ == "__main__":
    main()