# BeeMonitor — 蜂群進出辨識與行為監測

本研究實作了一個可部署於蜂巢出入口的 **智慧影像監測系統**。  
系統結合 **YOLOv12** 與 **ByteTrack**，能夠長期監測並自動辨識三類蜜蜂的進出行為：  

- 工蜂 (Worker)  
- 攜帶花粉工蜂 (Pollen)  
- 雄蜂 (Drone)  

系統可輸出事件紀錄、進出數據，並提供多目標追蹤 (MOT) 評估與數據分析。  

⚠️ 注意：
本 GitHub 專案並未包含完整的研究內容（如完整數據集、訓練流程與全部分析），而是整理出研究過程中核心的程式碼部分，提供參考或延伸使用。

---

## 🌐 自動化蜂群資料分析系統（GitHub + GitHub Actions）

本研究另實作一套 **自動化分析模組**，將邊緣運算設備（TX2）蒐集到的蜂群監測數據，自動同步到雲端並進行定期分析與可視化。

### 功能特色

- 透過 **Git 自動上傳** 蜂群監測 CSV 數據至 GitHub
- 使用 **GitHub Actions 定時排程** 執行 Python 分析程式
- 自動生成：
  - 蜂群活動趨勢圖
  - 蜂群健康指標評估
  - 授粉效率分析圖表

此模組使系統具備「**長期遠端監測 + 自動化報告產生**」能力，無需人工定期下載資料與手動分析。

### 專案程式碼

雲端自動化分析專案已獨立存放於：

🔗 https://github.com/Avidityyds/beeMonitorAnalysis

該專案包含：

- GitHub Actions workflow 設定
- Python 資料分析與繪圖程式
- 數據結構與檔案格式說明

---

## 硬體 / 執行環境

- **NVIDIA Jetson TX2** 或其他具備 GPU 的邊緣運算設備  
- 建議使用 **Docker** 部署，以避免 TX2 上因 JetPack / CUDA / PyTorch 版本不同造成的相依問題  
  👉 [Ultralytics 官方 Jetson Docker 指南](https://docs.ultralytics.com/guides/nvidia-jetson/#jetpack-support-based-on-jetson-device)

- 攝影機：USB Camera (Logitech C922 Pro 測試)  

軟體需求：
- Python 3.9+
- Ultralytics YOLO (YOLOv12n)
- OpenCV
- ByteTrack
- motmetrics
- Jupyter Notebook (for data analysis)

---

## 📂 專案結構
```text
BeeMonitor/
├── src/
│   ├── realtime/
│   │   └── bee_counter_combined.py   # 即時部署用：蒐集蜂群進出事件
│   │                                  # 輸入：攝影機即時影像 + YOLO模型權重檔（.pt）
│   │                                  # 輸出：CSV (進出事件 + 蜂種分類)
│   │
│   ├── infer/
│   │   └── inference.py              # 影片檔案推論：進出計數
│   │                                  # 輸入：YOLO模型權重檔（.pt）+ 影片檔
│   │                                  # 輸出：統計結果、推論影片
│   │
│   ├── MOT/
│   │   ├── getPredict.py             # 轉換推論結果為 MOT 格式 (MOTChallenge)
│   │   │                              # 輸出：.txt（格式：frame, id, x, y, w, h, score, -1, -1, -1）
│   │   └── MOTtest.py                # MOT 指標測試：與 Ground Truth 比對（Ground Truth 需自行透過 CVAT 標註影片並匯出）
│   │                                  # 輸出：MOTA、IDF1、Precision、Recall
│   │
│   └── dataAnalysis/                 # 分析與繪圖
│       └── DrawFig.ipynb              # 輸入：bee_counter_combined.py 產生的 CSV
│                                      # 輸出：統計圖表 (進出量 / 花粉率 / 蜂種比例)
│                                      
│
├── README.md                         # 專案說明
└── LICENSE                           # 授權條款
