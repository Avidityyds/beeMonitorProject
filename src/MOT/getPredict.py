import cv2
from ultralytics import YOLO
import numpy as np

# 載入模型
model = YOLO('/Users/jameschen/Desktop/BeeProject/BeeProject3class_exp2/best.pt')  # 模型路徑
video_path = '/Users/jameschen/Desktop/BeeProject/MOTtest/beeVideoMOTtest.mp4'  # 影片路徑

# 打開影片
cap = cv2.VideoCapture(video_path)

# 打開寫入 txt 檔
with open('pred.txt', 'w') as f:
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # 用 YOLO 做追蹤
        results = model.track(
            frame,
            conf=0.678,  # 最佳信心閥值（from F1 curve）
            persist=True,  # 讓它保持追蹤 ID
            tracker="bytetrack.yaml"
        )

        # 確認有偵測到物件
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()  # 取得 (x_center, y_center, w, h)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # ID
            confs = results[0].boxes.conf.cpu().numpy()  # 信心值
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # 類別

            for box, track_id, conf, class_id in zip(boxes, track_ids, confs, class_ids):
                # YOLO 是 xywh (中心點)，MOT 需要的是 (x1, y1, w, h)
                x_center, y_center, w, h = box
                x1 = x_center - w / 2
                y1 = y_center - h / 2

                # 寫入 MOT 格式：
                # frame, id, x, y, w, h, conf, class, visibility
                f.write(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},{class_id},-1\n")

# 關閉影片
cap.release()

print("預測結果已經寫入 pred.txt")
