# LayoutHub

LayoutHub 是一個整合了室內 360° 全景圖 Layout 估計、可視化與生成的框架。

## 功能特色

1.  **推論 (Inference)**: 整合多種模型進行 Layout 預測。
2.  **2D 可視化**: 在全景圖上繪製預測的 Layout 邊界 (Visualize Layout)。
3.  **3D 可視化**: 將 Layout 轉換為 3D Mesh 並進行互動式檢視 (Visualize 3D)。
4.  **平面圖生成**: 生成包含全景視角與俯視平面圖的綜合視圖 (Generate Floor Plan)。
5.  **輔助工具**: 包含 PLY 模型檢視與佈局幾何修復工具。

## 目錄結構

*   `run.py`: 執行模型推論的主程式。
*   `visualize_corners.py`: 2D Layout 可視化腳本 (在圖片上畫線)。
*   `visualize_3d.py`: 3D Layout 可視化與 PLY 生成腳本。
*   `visualize_combined.py`: 生成全景圖 + 俯視平面圖的綜合視圖。
*   `view_ply.py`: 簡單的 3D PLY 檔案檢視器。
*   `rectify_layout.py`: 佈局幾何修復工具 (Rectification)。
*   `models/`: 模型定義。
*   `utils/`: 幾何運算工具 (包含 `panostretch.py`, `post_proc.py`)。

## 使用方法

### 1. 推論 (Inference)

```bash
python run.py --model <model_name> --mode infer --img_glob <path_to_images> --output_dir <output_path>
```
*   支援模型: `horizonnet`, `hohonet`, `dmhnet`, `lgtnet` (視 `models/` 支援而定)

### 2. 2D 可視化 (Visualize Layout)

在全景圖上繪製佈局邊界：

```bash
python visualize_corners.py --layout <layout.json/txt> --image <image.jpg> --out <output.png>
```

### 3. 3D 可視化 (Visualize 3D)

互動式檢視 3D 房間模型，或輸出 PLY 檔案：

```bash
# 開啟視窗檢視
python visualize_3d.py --img <image.jpg> --layout <layout.json> --vis

# 輸出 PLY 檔案
python visualize_3d.py --img <image.jpg> --layout <layout.json> --out output.ply
```

### 4. 生成平面圖 (Generate Floor Plan)

生成包含全景與頂視圖的圖片：

```bash
python visualize_combined.py --layout <layout.json> --image <image.jpg> --out <combined.png>
```

### 5. 其他工具

**檢視 PLY 檔案:**
```bash
python view_ply.py <file.ply>
```

**佈局幾何修復 (Rectify):**
將佈局旋轉並對齊至 Manhattan World 座標系：
```bash
python rectify_layout.py <input.json> <output.json>
```

## 需求環境

*   Python 3.x
*   NumPy
*   OpenCV (`opencv-python`)
*   Open3D (`open3d`)
*   Pillow (`PIL`)
*   SciPy
