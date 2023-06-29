import cv2
import numpy as np


class opticalTracker:
    def __init__(self):
        initial_ids = []
        self.trackers = {id: {'point': None, 'prev_gray': None, 'prev_point': None, 'speed': None} for id in
                         initial_ids}

    def compute_velocity(self, gray, id):
        if self.trackers[id]['prev_gray'] is not None:
            new_point, status, _ = cv2.calcOpticalFlowPyrLK(self.trackers[id]['prev_gray'], gray,
                                                            np.array([self.trackers[id]['prev_point']],
                                                                     dtype=np.float32),
                                                            np.array([self.trackers[id]['point']],
                                                                     dtype=np.float32))
            if status[0][0]:
                dx = new_point[0][0] - self.trackers[id]['point'][0]
                dy = new_point[0][1] - self.trackers[id]['point'][1]

                speed = np.sqrt(dx ** 2 + dy ** 2)
                self.trackers[id]['speed'] = speed
                self.trackers[id]['prev_point'] = self.trackers[id]['point']
                self.trackers[id]['point'] = new_point[0]  # 找新的特征点
            else:
                self.trackers[id]['prev_point'] = self.trackers[id]['point']
                self.trackers[id]['point'] = None  # speed不变，将point置为None，后续update更新
            self.trackers[id]['prev_gray'] = gray
        else:
            self.trackers[id]['prev_gray'] = gray
            self.trackers[id]['prev_point'] = self.trackers[id]['point']

    def update_trackers(self, tracked_boxes, boxes_id, gray):
        # 如果车辆已经不在画面中，删除对应的tracker
        for id in list(self.trackers.keys()):
            if id not in boxes_id:
                del self.trackers[id]
        # 在画面中
        for box, id in zip(tracked_boxes, boxes_id):
            if id not in self.trackers:  # 为新出现的车辆创建新的tracker
                x = (float(box[0]) + float(box[2])) / 2
                y = (float(box[1]) + float(box[3])) / 2
                center = np.array((x, y), dtype=np.float32)
                self.trackers[id] = {'point': center, 'prev_gray': None, 'prev_point': None, 'speed': None}
            elif self.trackers[id]['point'] is None:  # 上一次特征点寻找失败
                x = (float(box[0]) + float(box[2])) / 2
                y = (float(box[1]) + float(box[3])) / 2
                center = np.array((x, y), dtype=np.float32)
                self.trackers[id]['point'] = center

            self.compute_velocity(gray, id)
        return self.trackers


# 初始化
optical_tracker = opticalTracker()
