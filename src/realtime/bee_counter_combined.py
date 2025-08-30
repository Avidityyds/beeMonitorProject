import cv2
import numpy as np
import os
import csv
import time
import datetime
import pytz  
from collections import defaultdict
from shapely.geometry import LineString, Point
from ultralytics import YOLO

class ObjectCounter:
    def __init__(self, class_names, reg_pts, line_thickness=2, draw_tracks=True,
                 conf_thresholds=None):
        self.class_names = class_names
        self.reg_pts = reg_pts
        self.counts = {name: {'in': 0, 'out': 0} for name in class_names.values()}
        self.tracks = defaultdict(list)
        self.crossed = set()
        self.line = LineString(reg_pts)
        self.frame_count = 0
        self.total_inference_time = 0
        self.fps_list = []
        self.precision_list = []
        self.line_thickness = line_thickness
        self.draw_tracks = draw_tracks
        self.conf_thresholds = conf_thresholds or {
            0: 0.6,
            1: 0.5,
            2: 0.6
        }

    def draw_box(self, img, box, class_id, confidence):
        if confidence < self.conf_thresholds.get(class_id, 0.5): 
            return
        
        x1, y1, x2, y2 = map(float, box)
        label = self.class_names.get(class_id, "unknown")

        color_map = {
            0: (255, 0, 255),
            1: (0, 128, 255),
            2: (0, 0, 255)
        }
        color = color_map.get(class_id, (255, 255, 255))

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, self.line_thickness)
        cv2.putText(img, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=1, lineType=cv2.LINE_AA)

    def start_counting(self, frame, results):
        self.frame_count += 1
        cv2.line(frame, self.reg_pts[0], self.reg_pts[1], (0, 255, 0), self.line_thickness)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            avg_precision = np.mean(confidences) if len(confidences) > 0 else 0
            self.precision_list.append(avg_precision)

            for box, track_id, conf, class_id in zip(boxes, track_ids, confidences, class_ids):
                self.draw_box(frame, box, class_id, conf)
                center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                self.tracks[track_id].append(center)

                if len(self.tracks[track_id]) >= 2 and track_id not in self.crossed:
                    old_point = Point(self.tracks[track_id][-2])
                    new_point = Point(self.tracks[track_id][-1])
                    if self.line.crosses(LineString([old_point, new_point])):
                        self.crossed.add(track_id)
                        direction = 'in' if new_point.y < old_point.y else 'out'
                        class_name = self.class_names.get(class_id, 'unknown')
                        self.counts[class_name][direction] += 1
        return frame

def save_counts_to_csv(csv_path, counter, tag):
    tz = pytz.timezone('Asia/Taipei')  
    now_time = datetime.datetime.now(tz)
    record_data = {
        'dt': now_time.strftime("%Y-%m-%d %H:%M:%S"),
        'day': now_time.day,
        'hour': now_time.hour,
        'minute': now_time.minute,
        'tag': tag
    }
    for class_name, count in counter.counts.items():
        record_data[f"in_{class_name}"] = count['in']
        record_data[f"out_{class_name}"] = count['out']
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record_data)

def main():
    tz = pytz.timezone('Asia/Taipei')  
    model_path = '/workspace/BeeCount/best.pt'
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    class_names = {0: "worker", 1: "pollen", 2: "drone"}
    counter = ObjectCounter(
        class_names=class_names,
        reg_pts=[(0, height // 2), (width, height // 2)]
    )
    tag = "TX2_6"
    data_dir = "bee_data"
    os.makedirs(data_dir, exist_ok=True)

    now_time = datetime.datetime.now(tz)
    csv_path = os.path.join(data_dir, f"{now_time.strftime('%Y-%m')}_{tag}_inout.csv")
    record_interval = datetime.timedelta(minutes=5)
    record_start_time = datetime.datetime.now(tz)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            counter.start_counting(frame, results)

            if datetime.datetime.now(tz) - record_start_time > record_interval:
                save_counts_to_csv(csv_path, counter, tag)
                counter.counts = {k: {'in': 0, 'out': 0} for k in counter.counts}
                record_start_time = datetime.datetime.now(tz)

            # 長期執行把視窗關掉才不耗資源
            # cv2.imshow("Bee Counter", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
