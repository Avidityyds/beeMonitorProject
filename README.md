# BeeMonitor — 蜂群進出辨識與行為監測

## 介紹

本研究實作了一個可部署於蜂巢出入口的**智慧影像監測系統**，結合 YOLOv12 與 ByteTrack，能夠長期監測自動辨識並記錄 **工蜂 (Worker)**、**攜帶花粉工蜂 (Pollen)**、**雄蜂 (Drone)** 三類蜜蜂的進出行為，並提供多目標追蹤 (MOT) 評估與數據分析。  

⚠️ 注意：  
本 GitHub 專案並未包含完整的研究內容（如完整數據集、訓練流程與全部分析），而是整理出研究過程中**核心的程式碼部分**，提供參考或延伸使用。  

專案中的所有程式統一放在 `src/` 目錄下，以下說明各檔案的用途：


---

## 📂 src/realtime/
- **bee_counter_combined.py**  
  - 即時部署用的數據蒐集程式  
  - **需要：**  
    - 攝影機（USB）  
    - 已訓練好的 YOLO 模型權重檔（`.pt`）
    - 若要在 TX2 上執行需透過 **Docker** 執行，請參考 Ultralytics 官方指引：  
      👉 https://docs.ultralytics.com/guides/nvidia-jetson/#jetpack-support-based-on-jetson-device
  - **輸出：**  
    - CSV（紀錄進/出事件與蜂種分類）  
  - 這些 CSV 可供 `DrawFig.ipynb` 做後續分析  

---

## 📂 src/infer/
- **inference.py**  
  - 影片進出計數程式  
  - **需要：**  
    - YOLO 模型權重檔（`.pt`）  
    - 輸入影片檔（`.mp4` 或其他格式）  
  - **輸出：**  
    - 統計結果（進巢 / 出巢數量）  
    - 推論結果影片

---

## 📂 src/MOT/
- **getPredict.py**  
  - 將推論結果轉換成 **MOT format**（MOTChallenge 規範）的 txt 檔  
  - **需要：**  
    - YOLO 模型權重檔（`.pt`）  
    - 輸入影片檔（`.mp4`）  
  - **輸出：**  
    - `*.txt`（格式：`frame, id, x, y, w, h, score, -1, -1, -1`）  

- **MOTtest.py**  
  - 將 `getPredict.py` 產生的 MOT format 結果與 **Ground Truth** 做比對  
  - **需要：**  
    - `getPredict.py` 輸出的 MOT txt  
    - Ground Truth MOT txt（需自行透過 **CVAT** 標註影片並匯出）  
  - **輸出：**  
    - MOTA、IDF1、Precision、Recall 等追蹤評估指標  

---

## 📂 src/dataAnalysis/
- **DrawFig.ipynb**  
  - Jupyter Notebook，用來做數據分析與繪圖  
  - **需要：**  
    - `bee_counter_combined.py` 產生的 CSV  
  - **輸出：**  
    - 統計圖表（進出量、花粉率、蜂種比例等）  

---
