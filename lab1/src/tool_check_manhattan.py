import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from pathlib import Path
import sys
import json

# 將 LayoutHub 加入路徑以便匯入 np_coor2xy
_LAYOUTHUB = Path(__file__).parent.parent.parent / 'LayoutHub'
if _LAYOUTHUB.exists():
    sys.path.insert(0, str(_LAYOUTHUB))

try:
    from utils.geom import rectify_polygon
except ImportError:
    pass

try:
    from utils.post_proc import np_coor2xy
except ImportError:
    print("Error: Could not import LayoutHub utils.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Check Manhattan Assumption for a Room Layout")
    parser.add_argument('--layout_txt', required=True, help='Path to layout .txt file')
    parser.add_argument('--out', default='manhattan_check.png', help='Path to output visualization file')
    parser.add_argument('--out_json', default='', help='Optional Path to output checking stats into JSON format')
    return parser.parse_args()

def load_layout_txt(txt_path, pano_w=1024, pano_h=512, z=50):
    """
    讀取 TXT 檔中的全景像素座標，並使用 np_coor2xy 投影到真實的 3D 地板空間。
    跟 tool_view_corners.py 完全一樣。
    """
    path = Path(txt_path)
    if not path.exists():
        print(f"[錯誤] 找不到檔案: {txt_path}")
        return None
        
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])
            
    if len(pts) < 2:
        return None

    if len(pts) % 2 == 0:
        floor_pts = [pts[i] if pts[i][1] > pts[i+1][1] else pts[i+1] for i in range(0, len(pts), 2)]
    else:
        floor_pts = pts

    floor_pixel = np.array(floor_pts)
    
    floor_xy = np_coor2xy(floor_pixel, z=z,
                          coorW=pano_w, coorH=pano_h,
                          floorW=pano_w, floorH=pano_w)
                          
    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]
    try:
        floor_xy = rectify_polygon(floor_xy)
    except Exception:
        pass
        
    return floor_xy

def check_manhattan(xy):
    """
    計算每相鄰兩道牆壁間的夾角，
    曼哈頓假設代表每一道牆的夾角都必須是非常接近 90 度 (或 -90, 270)。
    """
    if len(xy) < 3:
        return []

    # 封閉多邊形
    shifted_xy = np.roll(xy, -1, axis=0) # 下一個角點
    
    # 邊的向量
    dx = shifted_xy[:, 0] - xy[:, 0]
    dy = shifted_xy[:, 1] - xy[:, 1]
    
    # 計算每個邊的角度 (rad)
    edge_angles = np.arctan2(dy, dx)
    
    # 兩相鄰邊之間的轉向角 (下一個邊的角度 - 當前邊的角度)
    shifted_angles = np.roll(edge_angles, -1)
    angle_diffs = shifted_angles - edge_angles
    
    # 限制在 [-pi, pi] 之間
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    
    # 轉為度數
    angle_diffs_deg = np.degrees(angle_diffs)
    
    # 偏離 90 度的誤差 (如果是曼哈頓，此值應該要是 0)
    # 取除以 90 的餘數，然後取距離 0 或 90 最短的差距 (因為有可能是 180 度，但這不影響垂直判斷)
    deviations = []
    
    print("\n--- 曼哈頓假設檢查結果 ---")
    for i, diff_deg in enumerate(angle_diffs_deg):
        # 取絕對值的除以 90 餘數
        remainder = abs(diff_deg) % 90
        # 距離 90 度的倍數多遠 
        dev = min(remainder, 90 - remainder)
        deviations.append(dev)
        
        corner_idx = (i + 1) % len(xy)  # 這是這兩條邊交會的那個角
        print(f"角 {corner_idx+1} (Corners {i+1}->{corner_idx+1}->{(corner_idx+1)%len(xy)+1}) "
              f"轉向角: {diff_deg:8.2f}° | 曼哈頓誤差: {dev:5.2f}°")
              
    mean_error = np.mean(deviations)
    max_error = np.max(deviations)
    print(f"\n✅ 平均角度誤差: {mean_error:.2f}°")
    print(f"✅ 最大角度誤差: {max_error:.2f}°")
    
    is_manhattan = max_error <= 5.0 # 五度以內視為符合
    if is_manhattan:
        print("🎉 結論: 該房間 **符合** 曼哈頓假設 (誤差極小)！")
    else:
        print("⚠️ 結論: 該房間 **不完全符合** 曼哈頓假設 (可能有斜牆或歪斜)。")
        
    return deviations, angle_diffs_deg

def main():
    args = parse_args()
    
    # 確保輸出的圖有字體支援
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    floor_xy = load_layout_txt(args.layout_txt)
    if floor_xy is None or len(floor_xy) < 3:
        print("Error: Could not parse valid floor polygon from TXT.")
        return
        
    # 計算並且列印曼哈頓檢查報告
    deviations, diffs_deg = check_manhattan(floor_xy)
    
    # 畫圖視覺化
    poly_closed = np.vstack([floor_xy, floor_xy[0]])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Manhattan Assumption Check", fontsize=16)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 畫出形狀
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], color='blue', linewidth=3)
    
    # 在每一個角旁標上它的誤差
    for i, (x, y) in enumerate(floor_xy):
        dev = deviations[(i - 1) % len(floor_xy)]  # 該角點是前一個 diff 的對應交點
        
        ax.plot(x, y, 'ro', markersize=8)
        
        # 如果誤差大於 5 度，用紅色標示，否則用綠色
        color = 'red' if dev > 5.0 else 'green'
        
        text = f" C{i+1}\nError:{dev:.1f}°"
        ax.text(x, y, text, fontsize=12, fontweight='bold', color=color)
        
    # 相機原點
    ax.scatter([0], [0], marker='^', color='black', s=150, label='Camera (0,0)')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"\n視覺化圖片已儲存至: {args.out}")
    
    if args.out_json:
        # Create dictionary of the stats
        stats_data = {
            "corners": [],
            "mean_error": float(np.mean(deviations)),
            "max_error": float(np.max(deviations)),
            "is_manhattan": bool(np.max(deviations) <= 5.0)
        }
        
        for i, (x, y) in enumerate(floor_xy):
            corner_id = i + 1
            # (i - 1) % len is because the deviation corresponding to point `i` relates to edges around it
            diff_deg = diffs_deg[(i - 1) % len(floor_xy)]
            dev = deviations[(i - 1) % len(floor_xy)]
            
            stats_data["corners"].append({
                "id": corner_id,
                "x_m": float(x),
                "y_m": float(y),
                "turn_angle_deg": float(diff_deg),
                "manhattan_error_deg": float(dev)
            })
            
        with open(args.out_json, "w") as f:
            json.dump(stats_data, f, indent=4)
        print(f"JSON 報告已儲存至: {args.out_json}")

if __name__ == "__main__":
    main()
