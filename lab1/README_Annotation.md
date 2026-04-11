## tool_view_corners.py

**功能：** 程式會彈出這個房間的 2D 俯視圖，並且在每一個牆角和門框上標註「數字 ID」

```jsx
python src/tool_view_corners.py --scene_dir data/group/58472_Floor1 --room_id <輸入房間ID>
```

## tool_edit_layout.py

```jsx
python src/tool_edit_layout.py \    --image "image.png" \    --layout "layout.txt" \    --out "新_layout.txt"
```

1. ****[滑鼠左鍵] 新增點位 (Add Point)****

2. ****[滑鼠右鍵] 復原 (Undo)****

3. ****[滑鼠中鍵] 或 [Enter 鍵] 儲存並結束****

## perfect_matches.json

將你找到的所有配對關係，寫入該樓層資料夾下的 `perfect_matches.json` 中。

```jsx
[
  {
    "src": "LivingRoom_ID.txt",
    "dst": "Kitchen_ID.txt",
    "idx_src": [4, 5], 
    "idx_dst": [1, 2],
  },
  {
    "src": "PrimaryBedroom_ID.txt",
    "dst": "Bathroom_ID.txt",
    "idx_src": [8, 9, 10], 
    "idx_dst": [3, 2, 1], 
  }
]
```

**參數說明：**

- `src` / `dst`: 兩個相鄰房間的 txt 檔名（不需要加路徑，只需給檔名即可）。
- `idx_src`: 房間 src 的特徵點 ID 陣列（`tool_view_corners.py` 畫出來的數字）。
- `idx_dst`: 房間 dst 的特徵點 ID 陣列，長度必須與 idx_src 相同，且順序對應。

---

## 🚀 標註完成後：如何產生完美的樓層平面圖

當你把整層樓的配對關係都寫入 `perfect_matches.json` 之後，請依序執行以下三個步驟來生成最終結果：

### Step 1: 產生高置信度的邊界約束 (GTSAM Edges)
這會將你記錄的點陣列轉換成小誤差的旋轉與平移約束：
```bash
python src/tool_generate_gtsam_edges.py \
    --matches data/group/58472_Floor1/perfect_matches.json \
    --layout_dir data/group/58472_Floor1/layout_gt/ \
    --out data/group/58472_Floor1/perfect_edges.json
```

### Step 2: 進行全域位姿圖優化 (Pose Graph Optimization)
讓 GTSAM 自動推算所有房間的完美對齊相對旋轉角度（解決累積誤差）：
```bash
python src/03_optimize_pose_graph_gtsam.py \
    --edges data/group/58472_Floor1/perfect_edges.json \
    --init data/group/58472_Floor1/initial_poses.json \
    --out data/group/58472_Floor1/perfect_poses.json \
    --report data/group/58472_Floor1/perfect_report.json
```

### Step 3:對齊與渲染 (Perfect Tree Snapping)

```bash
python src/08_perfect_tree_snapping.py \
    --scene_dir data/group/58472_Floor1 \
    --poses data/group/58472_Floor1/perfect_poses.json \
    --matches data/group/58472_Floor1/perfect_matches.json \
    --out viz_output/58472_Floor1_Final.png
```
