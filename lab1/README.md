# Multi-room Spatial Alignment & Global Optimization
> A Pose Graph Approach for 3D Floorplan Reconstruction

## 📝 專案簡介 (Project Overview)
本專案旨在解決單目 3D 房間重建（如 HorizonNet）在進行「多房間拼接 (Multi-room Stitching)」時，因預測誤差所導致的累積漂移、房間旋轉扭曲與重疊問題。

本系統捨棄傳統不可靠的相機位移測量值，改採**幾何特徵匹配 (Geometric Feature Matching)** 獲取精確的局部空間約束，並引入 **GTSAM (Pose Graph Optimization)** 作為後端全域最佳化，最終將多個局部座標系的單一房間，精準對齊並渲染為全域一致的完整樓層平面圖。

## ✨ 核心技術 (Key Features)
* **特徵級空間對齊 (Spatial Alignment):** 透過相鄰房間共用門框或牆角的特徵匹配，計算完美的 $SE(2)$ 相對位姿。
* **曼哈頓世界轉正 (Manhattan Alignment):** 解決房間初始預測時的傾斜與旋轉問題，將所有房間強制對齊正交網格。
* **全域位姿圖最佳化 (Global Pose Graph Optimization):** 結合 GTSAM 消除前端里程計的累積誤差 (Accumulated Drift)。

---

## 📂 程式碼架構與模組說明 (Repository Structure)

本專案的原始碼位於 `src/` 目錄下，主要分為「主幹管線 (Main Pipeline)」與「特徵開發工具 (Custom Tools)」兩大類：

### 1. 主幹管線 (Main Pipeline)
這些腳本構成了一個端到端 (End-to-End) 的自動化處理流程：

* **`00_build_manifest.py`**: 資料前處理，建構讀取房間幾何與影像資訊的清單檔。
* **`01_build_edges_from_hotspots.py`**: [Baseline] 原始基線測試，從預測的熱點 (Hotspots) 中萃取相對位姿（通常帶有高誤差）。
* **`02_init_poses_bfs.py`**: 利用廣度優先搜尋 (BFS) 走訪位姿圖，為所有房間產生初始的絕對全域座標。
* **`03_optimize_pose_graph_gtsam.py`**: **[核心後端]** 載入位姿圖與約束邊界，呼叫 GTSAM 進行全域最佳化，攤平累積誤差。
* **`04_viz_pose_graph.py`**: 視覺化工具，將 GTSAM 最佳化前後的相機軌跡與拓樸網路繪製成圖表。
* **`05_draw_floorplan_overlay.py`**: **[最終渲染]** 讀取最佳化後的絕對座標與 2D 房間輪廓，繪製出無縫貼合的全樓層平面圖。
* **`06_estimate_theta_priors_from_layout.py`**: 針對各房間的幾何輪廓，利用連續向量 4 倍角公式計算主導角度，提供極其重要的先驗旋轉角度 (Theta Priors)，解決房間歪斜旋轉的問題。

### 2. 特徵開發與幾何運算工具 (Custom Tools)
這些工具用於手動特徵匹配與幾何數學驗證：

* **`tool_view_corners.py`**: 視覺化單一房間的角點與 Index，輔助人工尋找相鄰房間的對應點 (Correspondences)。
* **`tool_pairwise_verifier.py`**: 雙房間精準對齊測試，透過輸入兩個角點，嚴格計算出 $dx, dy, d\theta$ 並渲染拼接結果。
* **`tool_global_stitcher.py`**: 簡易版全域拼接器（未使用 GTSAM），用於觀察純前端拼接所產生的末端累積漂移現象。
* **`tool_generate_gtsam_edges.py`**: **[核心前端]** 將我們透過特徵匹配算出的完美局部關係，轉換為 GTSAM 可讀取的 `edges.json` 格式，作為高品質的約束條件。

### 3. 核心函式庫
* **`utils/`**: 包含各腳本共用的數學運算、幾何轉換與 I/O 輔助函式。

---

## 🚀 未來展望 (Future Work)
目前系統已具備強健的幾何拓樸骨架，接下來的升級計畫包含：
1. **影像與幾何融合 (Geometry-Vision Fusion):** 引入全景圖的視覺紋理對齊 (Photometric Alignment)，解決單目網路造成的尺度漂移 (Scale Drift) 與房間重疊問題。
2. **尺度最佳化 (Scale Optimization):** 將最佳化空間從剛體變換 ($SE(2)$) 升級為相似變換 ($Sim(2)$)。
3. **全自動特徵匹配:** 導入深度學習模型 (如 SuperGlue) 取代人工標記，實現全自動管線。