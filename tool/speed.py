import cv2
import numpy as np

from tool.optical_flow import optical_tracker


def pixel_to_real_speed(pixel_speed,  # 目标的像素速度，单位像素/帧。
                        ppm=15,  # 像素每米
                        fps=50):  # 视频的帧率，单位帧/秒。

    # 计算目标的实际速度（单位：米/秒）
    real_speed_m_per_s = pixel_speed * fps / ppm

    # 将速度转换为km/h
    real_speed_km_per_h = real_speed_m_per_s * 3.6
    return real_speed_km_per_h


def get_flow_pixel_speeds(tracked_boxes, boxes_ids, frame):  # 获取所有特征点的速度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()
    op_track = optical_tracker.update_trackers(tracked_boxes, boxes_ids, gray)
    speeds = []
    for id in boxes_ids:
        speed = op_track[id]['speed']
        if speed is None:
            speed = 0.0
        speeds.append(speed)
    return np.array(speeds)


old_boxes = None  # numpy
old_ids = None  # numpy
old_boxes_center = None  # (x,y),list


def get_boxes_center_pixel_speeds(new_boxes, ids, frame):
    new_boxes = np.array(new_boxes)
    ids = np.array(ids)
    global old_boxes, old_ids, old_boxes_center
    speeds = []
    center = []
    for (x1, x2, y1, y2) in new_boxes:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center.append((center_x, center_y))
    # 获取速度+更新
    if old_boxes is None:
        shape = len(new_boxes)
        speeds = np.full(fill_value=0.0, shape=shape)

        old_boxes = new_boxes
        old_boxes_center = center
        old_ids = ids
    else:
        for i in range(len(new_boxes)):
            index = np.where(old_ids == ids[i])
            if len(index[0]) == 0:
                speed = 0.0
            else:
                new_x, new_y = center[i]
                old_x, old_y = old_boxes_center[index[0][0]]
                dx = np.array(new_x - old_x)
                dy = np.array(new_y - old_y)
                speed = np.sqrt(dx ** 2 + dy ** 2)
            speeds.append(speed)
        speeds = np.array(speeds)

        old_boxes = new_boxes
        old_boxes_center = center
        old_ids = ids
    return speeds
