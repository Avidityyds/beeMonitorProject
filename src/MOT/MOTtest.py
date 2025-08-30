import motmetrics as mm
import pandas as pd

# 載入 ground truth
gt = pd.read_csv('/Users/jameschen/Desktop/BeeProject/MOTtest/groundTrue.txt', header=None)
gt.columns = ['FrameId', 'ObjectId', 'X', 'Y', 'Width', 'Height', 'Conf', 'ClassId', 'Vis']

# 載入預測結果 pred
pred = pd.read_csv('/Users/jameschen/Desktop/BeeProject/MOTtest/pred.txt', header=None)
pred.columns = ['FrameId', 'ObjectId', 'X', 'Y', 'Width', 'Height', 'Conf', 'ClassId', 'Vis']

# 初始化 accumulator
acc = mm.MOTAccumulator(auto_id=True)

# 獲取所有幀編號
frames = sorted(gt['FrameId'].unique())

for frame in frames:
    # 取出當前幀的 gt 和 pred
    gt_frame = gt[gt['FrameId'] == frame]
    pred_frame = pred[pred['FrameId'] == frame]

    # 提取 ID 列表
    gt_ids = gt_frame['ObjectId'].tolist()
    pred_ids = pred_frame['ObjectId'].tolist()

    # 提取 bbox（格式：[x, y, w, h]）
    gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
    pred_boxes = pred_frame[['X', 'Y', 'Width', 'Height']].values

    # 計算距離矩陣（IoU）
    distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5) # IOU可調整

    # 更新 accumulator
    acc.update(
        gt_ids,
        pred_ids,
        distances
    )

# 取得指標
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='結果')

# 顯示結果
pd.set_option('display.max_columns', None)
print(summary.transpose())
