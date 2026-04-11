# Multi-room Spatial Alignment & Global Optimization
> A Pose Graph Approach for 3D Floorplan Reconstruction

## 📝 專案簡介 (Project Overview)
本專案旨在解決單目 3D 房間重建（如 HorizonNet）在進行「多房間拼接 (Multi-room Stitching)」時，因預測誤差所導致的累積漂移、房間旋轉扭曲與重疊問題。




## 📂 程式碼架構與模組說明 (Repository Structure)

本專案的原始碼位於 `src/` 目錄下，主要分為「主幹管線 (Main Pipeline)」與「特徵開發工具 (Custom Tools)」兩大類：

### 1. 主幹管線 (Main Pipeline)
這些腳本構成了一個端到端 (End-to-End) 的自動化處理流程：

* **`00_build_manifest.py`**: 資料前處理，建構讀取房間幾何與影像資訊的清單檔。
* **`01_build_edges_from_hotspots.py`**: [Baseline] 原始基線測試，從預測的熱點 (Hotspots) 中萃取相對位姿（通常帶有高誤差）。
* **`02_init_poses_bfs.py`**: 利用廣度優先搜尋 (BFS) 走訪位姿圖，為所有房間產生初始的絕對全域座標。
* **`03_optimize_pose_graph_gtsam.py`**: **[核心後端]** 載入位姿圖與約束邊界，呼叫 GTSAM 進行全域最佳化，攤平累積誤差。
* **`04_viz_pose_graph.py`**: 視覺化工具，將 GTSAM 最佳化前後的相機軌跡與拓樸網路繪製成圖表。
* **`05_draw_floorplan_overlay.py`**: 讀取最佳化後的絕對座標與 2D 房間輪廓，繪製出初始的重疊樓層平面圖。
* **`06_snap_walls_and_draw.py`**: **[幾何後處理]** 在全域座標中執行強制曼哈頓對齊 (Global Axis-Alignment)，並根據距離閾值 (如 20cm) 自動吸附與合併相近的平行牆面。
* **`07_advanced_snapping.py`**: **[拓樸後處理]** 導入 Union-Find 叢集演算法，掃描並強制綁定小於 0.5m 距離的真實 90 度房間轉角，達成全域網格無縫化。
* **`08_perfect_tree_snapping.py`**: **[終極渲染]** 放棄 GTSAM 造成的滑動平移誤差，將配對圖視為無環樹 (Acyclic Graph)，使用 BFS 從根節點嚴苛平移，搭配 1D 牆壁吸附，輸出數學上保證 **0 縫隙、0 重疊** 的最終樓層平面圖。

### 2. 特徵開發與幾何運算工具 (Custom Tools)
這些工具用於手動特徵匹配與幾何數學驗證：

* **`tool_view_corners.py`**: 視覺化單一房間的角點與 Index，輔助人工尋找相鄰房間的對應點 (Correspondences)。
* **`tool_pairwise_verifier.py`**: 雙房間精準對齊測試，透過輸入兩個角點，嚴格計算出 $dx, dy, d\theta$ 並渲染拼接結果。
* **`tool_global_stitcher.py`**: 簡易版全域拼接器（未使用 GTSAM），用於觀察純前端拼接所產生的末端累積漂移現象。
* **`tool_generate_gtsam_edges.py`**: **[核心前端]** 將透過特徵匹配算出的完美局部關係轉換為 GTSAM 可讀取的 `edges.json` 格式，在此專案中我們並引入極端約束 (sigma=0.005) 來強制錨定門框對應點。
* **`tool_check_manhattan.py`**: 檢驗各房間座標是否完全符合曼哈頓正交假設（各牆角為 90 度或 270 度），並量化輸出偏差角度誤差。

### 3. 核心函式庫
* **`utils/`**: 包含各腳本共用的數學運算、幾何轉換與 I/O 輔助函式。

---

