import argparse
import numpy as np
import cv2
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 放進 LayoutHub 的路徑
_LAYOUTHUB = Path(__file__).parent.parent.parent / 'LayoutHub'
if _LAYOUTHUB.exists():
    sys.path.insert(0, str(_LAYOUTHUB))

try:
    from utils.geom import rectify_polygon, align_to_manhattan
except ImportError:
    pass

try:
    from utils.post_proc import np_coor2xy
    import utils.panorama as pano_utils
except ImportError:
    print("Error: Could not import LayoutHub utils.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TXT layout: Panorama + Top-Down Floor Plan")
    parser.add_argument('--layout_txt', required=True, help='Path to layout .txt file')
    parser.add_argument('--image', required=True, help='Path to original panoramic image')
    parser.add_argument('--out', required=True, help='Path to output visualization file (.png)')
    return parser.parse_args()

def load_layout_txt(txt_path, pano_w=1024, pano_h=512, z=50):
    """
    讀取 TXT 檔中的全景像素座標，並使用 np_coor2xy 投影到真實的 3D 地板空間。
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

    # 如果資料是交錯的 (天花板, 地板)，抓出 y 座標較大的(地板)
    if len(pts) % 2 == 0:
        floor_pts = [pts[i] if pts[i][1] > pts[i+1][1] else pts[i+1] for i in range(0, len(pts), 2)]
    else:
        floor_pts = pts

    floor_pixel = np.array(floor_pts)
    
    # 進行投影 (同 tool_pairwise_verifier.py)
    floor_xy = np_coor2xy(floor_pixel, z=z,
                          coorW=pano_w, coorH=pano_h,
                          floorW=pano_w, floorH=pano_w)
                          
    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]  # Y 軸反轉
    try:
        floor_xy = rectify_polygon(floor_xy)
    except Exception:
        pass
        
    return floor_xy

# align_to_manhattan 移至 utils/geom.py

def main():
    args = parse_args()
    
    # 1. 讀取全景圖片
    img_src = cv2.imread(args.image)
    if img_src is None:
        print(f"Error: Could not read image {args.image}")
        return
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    
    # 2. 讀取 TXT 座標
    floor_xy = load_layout_txt(args.layout_txt)
    if floor_xy is None or len(floor_xy) < 3:
        print("Error: Could not parse valid floor polygon from TXT.")
        return
        
    # 3. 🌟 套用曼哈頓對齊魔法
    floor_xy_aligned, _ = align_to_manhattan(floor_xy)
    
    # 將多邊形封閉 (頭尾相連) 以便畫圖
    poly_closed = np.vstack([floor_xy_aligned, floor_xy_aligned[0]])
    
    # 4. 開始畫圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # --- 左半邊：原圖 ---
    ax1.imshow(img_src)
    ax1.set_title("Original Panorama")
    ax1.axis('off')
    
    # --- 右半邊：對齊後的平面圖 ---
    ax2.set_title("Top-Down Floor Plan (TXT -> Manhattan Aligned)")
    ax2.axis('equal') # 確保 X 軸和 Y 軸比例一致，房間才不會變形
    ax2.grid(True, linestyle='--')
    
    # 畫出牆壁線條
    ax2.plot(poly_closed[:, 0], poly_closed[:, 1], color='blue', linewidth=3)
    
    # 畫出角點並標上數字 (方便你之後找 Correspondence 對應關係！)
    for i, (x, y) in enumerate(floor_xy_aligned):
        ax2.plot(x, y, 'ro', markersize=8)
        # +1 讓標籤顯示為 1, 2, 3, 4 而不是 0, 1, 2, 3，這樣更直覺
        ax2.text(x, y, f" {i + 1}", fontsize=14, fontweight='bold', color='red')
        
    # 畫出相機原點 (0,0)
    ax2.scatter([0], [0], marker='^', color='black', s=150, label='Camera (0,0)')
    ax2.legend()
    
    # 如果原本的座標系是 Y 往下為正，可以把 Y 軸反轉讓它符合一般平面圖視角
    # ax2.invert_yaxis() 
    
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved visualization to {args.out}")

if __name__ == "__main__":
    main()