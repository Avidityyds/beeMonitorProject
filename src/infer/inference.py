import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from shapely.geometry import LineString, Point
import time
import datetime

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

        # 個別類別的 confidence threshold
        self.conf_thresholds = conf_thresholds or {
            0: 0.5,  # worker
            1: 0.5,  # pollen
            2: 0.5   # drone
        }

    def draw_box(self, img, box, class_id, confidence):
        if confidence < self.conf_thresholds.get(class_id, 0.5):
            return

        x1, y1, x2, y2 = map(float, box)
        label = self.class_names.get(class_id, "unknown")

        # 顏色對應：worker=紫，pollen=橘，drone=紅
        color_map = {
            0: (255, 0, 255),    # 紫色 - worker
            1: (0, 128, 255),    # 橘色 - pollen
            2: (0, 0, 255)       # 紅色 - drone
        }
        color = color_map.get(class_id, (255, 255, 255))

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, self.line_thickness)
        cv2.putText(img, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness=1, lineType=cv2.LINE_AA)

    def start_counting(self, frame, results):
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]

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

        # 顯示計數結果
        y_offset = 30
        for class_name, count in self.counts.items():
            cv2.putText(frame, f"{class_name} in: {count['in']} out: {count['out']}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_offset += 30

        # FPS & precision 顯示
        fps = len(self.fps_list) / sum(self.fps_list) if self.fps_list else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        avg_precision = np.mean(self.precision_list) if self.precision_list else 0
        cv2.putText(frame, f"Avg Precision: {avg_precision:.3f}",
                    (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame


def main():
    MODEL_PATH = '/home/ubuntu/PinHsun/outputs/bee_exp1/weights/best.pt'
    VIDEO_PATH = '/home/ubuntu/PinHsun/BeeProject/beeVideo.mp4'

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    class_names = {0: "worker", 1: "pollen", 2: "drone"}

    counter = ObjectCounter(
        class_names=class_names,
        reg_pts=[(0, height // 2), (width, height // 2)],
        conf_thresholds={0: 0.5, 1: 0.6, 2: 0.4}
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'output_{timestamp}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml"
            )

            inference_time = time.time() - start_time
            counter.total_inference_time += inference_time
            counter.fps_list.append(inference_time)
            if len(counter.fps_list) > 30:
                counter.fps_list.pop(0)

            annotated_frame = counter.start_counting(frame, results)
            out.write(annotated_frame)

    except KeyboardInterrupt:
        print("\nStopping capture...")

    finally:
        print("\nFinal Statistics:")
        print(f"Total frames processed: {counter.frame_count}")
        print(f"Average inference time: {counter.total_inference_time / counter.frame_count:.3f} seconds")
        print(f"Average precision: {np.mean(counter.precision_list):.3f}")
        print("\nCounts:")
        for class_name, count in counter.counts.items():
            print(f"{class_name}: In={count['in']}, Out={count['out']}")

        cap.release()
        out.release()
        print(f"\nVideo saved as {output_path}")


if __name__ == "__main__":
    main()
