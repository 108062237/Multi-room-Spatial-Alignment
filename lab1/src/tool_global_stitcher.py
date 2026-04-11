import argparse
import json
import sys
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# LayoutHub の utils を読み込む (tool_pairwise_verifier.py と同じ方法)
_LAYOUTHUB = Path(__file__).resolve().parent.parent.parent / 'LayoutHub'
if _LAYOUTHUB.exists():
    sys.path.insert(0, str(_LAYOUTHUB))

try:
    from utils.geom import rectify_polygon, align_to_manhattan
except ImportError:
    pass

try:
    from utils.post_proc import np_coor2xy
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    print("[警告] 無法載入 utils.post_proc（LayoutHub 路徑找不到）。")

# ==========================================
# 1. 檔案讀取與基礎幾何處理
# ==========================================
def load_layout_txt(txt_path, pano_w=1024, pano_h=512, z=50):
    """
    讀取 TXT 格式並用 np_coor2xy 投影到真正的 3D 地板 XY 座標。
    與 tool_pairwise_verifier.py 完全一致的方法。
    """
    if not HAS_UTILS:
        raise ImportError("缺少 utils.post_proc，無法投影 TXT 座標！")

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

    # 交錯格式 (天花板, 地板, ...)：取 y 較大的那個為地板
    if len(pts) % 2 == 0:
        floor_pixel = []
        for i in range(0, len(pts), 2):
            p1, p2 = pts[i], pts[i + 1]
            floor_pixel.append(p1 if p1[1] > p2[1] else p2)
    else:
        floor_pixel = pts

    floor_pixel = np.array(floor_pixel)

    # 投影到真正的 3D 地板 XY 座標
    floor_xy = np_coor2xy(
        floor_pixel, z=z,
        coorW=pano_w, coorH=pano_h,
        floorW=pano_w, floorH=pano_w
    )

    center = pano_w / 2 - 0.5
    floor_xy[:, 0] -= center
    floor_xy[:, 1] -= center
    floor_xy[:, 1] = -floor_xy[:, 1]  # 翻轉 Y，與 verifier 一致
    try:
        floor_xy = rectify_polygon(floor_xy)
    except Exception:
        pass
    return floor_xy


# align_to_manhattan is moved to utils/geom.py

# ==========================================
# 2. 核心變換數學 (SE2 -> 3x3 矩陣)
# ==========================================
def get_transform_matrix(pA_start, pA_end, pB_start, pB_end):
    """
    計算 B 對齊到 A 的 3x3 齊次變換矩陣。
    不加 + math.pi，讓 B_start↔A_start，B_end↔A_end（與 verifier 一致）。
    """
    angA = math.atan2(pA_end[1] - pA_start[1], pA_end[0] - pA_start[0])
    angB = math.atan2(pB_end[1] - pB_start[1], pB_end[0] - pB_start[0])
    dtheta = angA - angB  # 不加 pi：B_start→B_end 與 A_start→A_end 同向

    c, s = math.cos(dtheta), math.sin(dtheta)
    rx = c * pB_start[0] - s * pB_start[1]
    ry = s * pB_start[0] + c * pB_start[1]
    dx = pA_start[0] - rx
    dy = pA_start[1] - ry

    return np.array([
        [c, -s, dx],
        [s,  c, dy],
        [0,  0,  1]
    ])


def apply_transform(xy, T_matrix):
    """將 2D 座標陣列套用 3x3 變換矩陣"""
    # 擴充為齊次座標 [x, y, 1]
    xy_homo = np.hstack([xy, np.ones((len(xy), 1))])
    # 矩陣相乘並取回 [x, y]
    transformed = (T_matrix @ xy_homo.T).T
    return transformed[:, :2]

# ==========================================
# 3. 主程式：拓樸解析與全圖渲染
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="全局簡易拼接器 (無 GTSAM)")
    parser.add_argument('--matches', required=True, help='輸入的配對檔案 (例: perfect_matches.json)')
    parser.add_argument('--layout_dir', required=True, help='房間 txt 佈局檔所在的資料夾路徑 (例: layout_gt/)')
    parser.add_argument('--out', default='final_full_floorplan.png', help='輸出的完整平面圖檔案')
    args = parser.parse_args()

    # === 設定檔案路徑 ===
    matches_json_path = args.matches
    base_dir = Path(args.layout_dir)
    out_file = args.out
    
    try:
        matches = json.load(open(matches_json_path, 'r'))
    except FileNotFoundError:
        print(f"❌ 找不到 JSON 檔案: {matches_json_path}")
        return

    # === 建立雙向圖結構 (Graph) ===
    graph = {}
    for edge in matches:
        src, dst = edge["src"], edge["dst"]
        idx_src, idx_dst = edge["idx_src"], edge["idx_dst"]
        
        if src not in graph: graph[src] = []
        if dst not in graph: graph[dst] = []
        
        # 建立雙向連接
        graph[src].append({"neighbor": dst, "my_idx": idx_src, "their_idx": idx_dst})
        graph[dst].append({"neighbor": src, "my_idx": idx_dst, "their_idx": idx_src})

    # === 讀取所有房間形狀，並將 Root 轉正 ===
    root_room = matches[0]["src"] # 以清單第一筆的 src 作為宇宙中心
    rooms_data = {}
    for room_file in graph.keys():
        try:
            poly = load_layout_txt(base_dir / room_file)
            if room_file == root_room:
                poly, _ = align_to_manhattan(poly)  # 只有起點房間需要對齊世界座標
            rooms_data[room_file] = poly
        except Exception as e:
            print(f"❌ 讀取房間 {room_file} 失敗: {e}")
            return

    # === 廣度優先搜尋 (BFS) 算出所有房間的絕對座標矩陣 ===
    global_transforms = {root_room: np.eye(3)} # Root 位於原點，無位移旋轉
    queue = [root_room]
    
    while queue:
        current = queue.pop(0)
        T_current_global = global_transforms[current]
        
        for edge in graph[current]:
            neighbor = edge["neighbor"]
            if neighbor not in global_transforms:
                poly_curr = rooms_data[current]
                poly_neigh = rooms_data[neighbor]
                
                # 自動減 1 (1-based to 0-based index)
                curr_s = poly_curr[edge["my_idx"][0] - 1]
                curr_e = poly_curr[edge["my_idx"][1] - 1]
                neigh_s = poly_neigh[edge["their_idx"][0] - 1]
                neigh_e = poly_neigh[edge["their_idx"][1] - 1]
                
                # 計算：將鄰居對齊到當前房間的相對矩陣
                T_relative = get_transform_matrix(curr_s, curr_e, neigh_s, neigh_e)
                
                # 核心矩陣相乘：鄰居全域位置 = 當前全域位置 x 相對位置
                T_neighbor_global = T_current_global @ T_relative
                
                global_transforms[neighbor] = T_neighbor_global
                queue.append(neighbor)

    # === 畫出全樓層平面圖 ===
    print(f"⏳ 正在渲染 {len(global_transforms)} 個房間...")
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_title("Full Floor Plan", fontsize=22, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 如果你發現產出的圖上下顛倒，可以把下面這行的註解拿掉
    # ax.invert_yaxis() 
    
    colors = plt.cm.get_cmap('tab10', max(10, len(rooms_data)))
    
    for i, (room_name, T_global) in enumerate(global_transforms.items()):
        poly_local = rooms_data[room_name]
        poly_global = apply_transform(poly_local, T_global)
        
        poly_closed = np.vstack([poly_global, poly_global[0]])
        
        # 畫出房間外框與半透明填充
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], linewidth=3, color=colors(i), label=room_name[:8])
        ax.fill(poly_global[:, 0], poly_global[:, 1], color=colors(i), alpha=0.15)
        
        # 畫出相機位置
        cam_global = apply_transform(np.array([[0.0, 0.0]]), T_global)[0]
        ax.scatter(cam_global[0], cam_global[1], marker='^', color=colors(i), s=150, edgecolors='black')
        ax.text(cam_global[0] + 0.15, cam_global[1] + 0.15, f"Cam {i}", fontsize=11, fontweight='bold', color='black')

    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    
    print("\n" + "="*50)
    print(f"🎉 全樓層拼接完成！一共成功拼合了 {len(global_transforms)} 個房間！")
    print(f"🖼️ 請查看輸出檔案: {out_file}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()