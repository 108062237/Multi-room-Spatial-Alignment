#!/usr/bin/env python3
"""
Interactive Layout Annotation Tool
Allows users to visually re-annotate the corners (Ceiling and Floor pairs) of a panorama.

Instructions:
1. Run with python lab1/src/tool_edit_layout.py --image <img_path> --layout <old_txt> --out <new_txt>
2. Left-Click to add points.
   (Format: Click Ceiling point first, then Floor point, then move to next corner...)
3. Right-Click to undo the last clicked point.
4. Middle-Click (or press Enter) to finish and save.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import matplotlib
# Do NOT use 'Agg' backend here because we need GUI interaction
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to panorama image")
    ap.add_argument("--layout", default="", help="Path to existing layout text file to show as reference")
    ap.add_argument("--out", required=True, help="Path to save the new layout coordinates")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[!] Image not found: {img_path}")
        sys.exit(1)

    img = plt.imread(img_path)
    img_height, img_width = img.shape[:2]
    
    # 預設儲存格式為 1024x512，計算縮放比例
    scale_w = img_width / 1024.0
    scale_h = img_height / 512.0

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.canvas.manager.set_window_title('Layout Editor')
    ax.imshow(img)
    ax.axis('on')

    # Display old points if provided
    if args.layout and Path(args.layout).exists():
        pts = []
        for line in Path(args.layout).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
        if pts:
            pts = np.array(pts)
            # 讀進來的 txt 是 1024x512 座標，需要放大以畫在原圖上
            up_x = pts[:, 0] * scale_w
            up_y = pts[:, 1] * scale_h
            
            ax.plot(up_x, up_y, 'ro', markersize=3, label='Old Points (Reference)', alpha=0.9)
            # connecting the pairs (Ceiling -> Floor)
            for i in range(0, len(up_x), 2):
                if i+1 < len(up_x):
                    ax.plot([up_x[i], up_x[i+1]], [up_y[i], up_y[i+1]], 'r--', alpha=0.5)
            ax.legend(loc='upper right')

    plt.title("Re-annotate layout corners\nLEFT-CLICK: Add Point | RIGHT-CLICK: Undo | MIDDLE-CLICK or ENTER: Finish")
    plt.tight_layout()

    print("\n" + "="*50)
    print(" INTERACTIVE MODE ACTIVATED ")
    print("="*50)
    print(" [LEFT CLICK]   Add a point.")
    print("                (Please mark Ceiling then Floor for each corner)")
    print(" [RIGHT CLICK]  Undo last point.")
    print(" [MIDDLE CLICK] Finish and Save (Combines with old points).")
    print(" [ENTER KEY]    Finish and Save.")
    print("="*50 + "\n")

    # Interactive blocking input mapping (Left=1, Middle=2, Right=3)
    clicks = plt.ginput(
        n=-1, 
        timeout=0, 
        show_clicks=True,
        mouse_add=matplotlib.backend_bases.MouseButton.LEFT,
        mouse_pop=matplotlib.backend_bases.MouseButton.RIGHT,
        mouse_stop=matplotlib.backend_bases.MouseButton.MIDDLE
    )

    if not clicks and not pts:
        print("[!] No existing points and no points clicked. Exiting.")
        sys.exit(0)

    # 1. 處理新點：將使用者的點強制合併成 (Ceiling, Floor) 垂直線對，並轉回 1024x512
    new_pairs = []
    if clicks:
        for i in range(0, len(clicks), 2):
            if i+1 < len(clicks):
                x1, y1 = clicks[i]
                x2, y2 = clicks[i+1]
                mean_x = (x1 + x2) / 2.0
                c_y = min(y1, y2) # 影像座標 Y 越小越上面 (天花板)
                f_y = max(y1, y2)
                new_pairs.append({
                    'x': mean_x / scale_w,
                    'cy': c_y / scale_h,
                    'fy': f_y / scale_h
                })
                
    # 2. 處理舊點
    old_pairs = []
    if 'pts' in locals() and len(pts) > 0:
        for i in range(0, len(pts), 2):
            if i+1 < len(pts):
                x1, y1 = pts[i, 0], pts[i, 1]
                x2, y2 = pts[i+1, 0], pts[i+1, 1]
                mean_x = (x1 + x2) / 2.0
                c_y, f_y = min(y1, y2), max(y1, y2)
                old_pairs.append({
                    'x': mean_x,
                    'cy': c_y,
                    'fy': f_y
                })

    # 3. 合併並以水平方位角 X 排序 (這樣新點會完美接在正確的牆壁順序中間)
    all_pairs = old_pairs + new_pairs
    all_pairs.sort(key=lambda p: p['x'])
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        for p in all_pairs:
            f.write(f"{p['x']:.2f} {p['cy']:.2f}\n")
            f.write(f"{p['x']:.2f} {p['fy']:.2f}\n")
            
    print(f"[OK] Successfully combined {len(old_pairs)} old corners with {len(new_pairs)} new corners.")
    print(f"[OK] Saved {len(all_pairs)*2} points total to {out_path}")

if __name__ == "__main__":
    main()
