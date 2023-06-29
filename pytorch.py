import os.path

import cv2
import torch

from LPRNet.LPRNet import build_lprnet
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from tool.draw_tool import draw_box, draw_text, draw_id_speed, draw_rec
from tool.process import get_lpr_input, get_lpr_outstr
from tool.speed import get_flow_pixel_speeds, pixel_to_real_speed, get_boxes_center_pixel_speeds
from tool.track_tool import track_objects
from ultralytics import YOLO

image_format = [".jpg", ".jepg"]
video_format = [".mp4"]
lpr_pt = r".\weights\LPRNet\mybestLPRNet2.pt"
if_yoloCar_pt = r".\weights\yoloCar\inaccurate_fast\yolo_for_car.pt"
if_yoloPlate_pt = r".\weights\yoloPlate\inaccurate_fast\yolo_for_plate.pt"
sa_yoloCar_pt = r".\weights\yoloCar\slow_accurate\yoloCar.pt"
sa_yoloPlate_pt = r".\weights\yoloPlate\slow_accurate\yolo_for_plate.pt"
mars_pb = r"./weights/mars/mars.pb"
fst_yoloCar_pt = r".\weights\yoloCar\fastest\best.pt"

metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)

# Initialize the tracker
tracker = Tracker(metric)

# Process multiple frames
# for frame, boxes in video:
#     tracked_boxes,ids = track_objects(tracker, frame, boxes)

import numpy as np
import torch.nn as nn


def build_model(ptPath: str, model_type: str):
    if model_type == "lprnet":
        model = build_lprnet(phase="val")
        model.load_state_dict(torch.load(ptPath))
        model.cuda()
    elif model_type == "yolov8":
        model = YOLO(ptPath).model.cuda()
    else:
        print("error type")
        exit(-1)
    return model


def run_model(model: nn.Module, input_data: np.ndarray):
    device = torch.device("cuda")
    model.to(device)
    input_data = torch.tensor(input_data, dtype=torch.float32, device=device)
    output = model.forward(input_data)
    if type(output) == tuple:
        output = output[0]
    return output.detach().cpu().numpy()


def run_plate(yoloPlate, lpr_path, source, plate_label):
    yolo = build_model(yoloPlate, "yolov8")
    lpr = build_model(lpr_path, "lprnet")
    if os.path.splitext(source)[1] in video_format or isinstance(source, int):
        cap = cv2.VideoCapture(source)
        while cap.isOpened():  # 检查是否成功初始化，否则就 使用函数 cap.open()
            # Capture frame-by-frame  逐帧从摄像头中读取图像
            ret, frame = cap.read()  # ret 返回一个布尔值 True/False
            frame = cv2.resize(frame, (640, 640))  # 640,640,3
            inputs = np.array([frame]).transpose((0, 3, 1, 2)).astype(float)  # 1,3,640,640
            inputs /= 255
            outputs = run_model(yolo, inputs)
            new_img, final_boxes, final_classes, final_confidences = draw_box(frame, outputs, False, plate_label)
            plate_str_list = get_lpr_outstr(run_model(lpr, get_lpr_input(new_img, final_boxes)))
            new_img = draw_text(new_img, final_boxes, plate_str_list)
            cv2.imshow('frame', new_img)
            cv2.setWindowTitle('frame', 'result')
            key = cv2.waitKey(delay=5)
            if key == ord("q"):
                break
    elif os.path.splitext(source)[1] in image_format:
        image = cv2.imread(source, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (640, 640))
        inputs = np.array([image]).transpose((0, 3, 1, 2)).astype(float)
        inputs /= 255
        outputs = run_model(yolo, inputs)
        new_img, final_boxes, final_classes, final_confidences = draw_box(image, outputs, False, plate_label, True)
        plate_str_list = get_lpr_outstr(run_model(lpr, get_lpr_input(new_img, final_boxes)))
        new_img = draw_text(new_img, final_boxes, plate_str_list)
        cv2.imshow('frame', new_img)
        cv2.setWindowTitle('frame', 'result')
        cv2.waitKey(0)
    else:
        print(f"error format {os.path.splitext(source)[1]}")


def run_car(yoloCar, source, car_label: int, need_speed: bool = True, pixel_speed_function=None,
            need_track: bool = True):
    yolo = build_model(yoloCar, "yolov8")
    if os.path.splitext(source)[1] in video_format or isinstance(source, int):
        image_list = []
        cap = cv2.VideoCapture(source)
        if need_speed:
            need_track = True
        signals=0
        while cap.isOpened():  # 检查是否成功初始化，否则就 使用函数 cap.open()
            # Capture frame-by-frame  逐帧从摄像头中读取图像
            ret, frame = cap.read()  # ret 返回一个布尔值 True/False
            if not ret:
                break
            h, w = frame.shape[0], frame.shape[1]  # 记录原图像大小
            frame = cv2.resize(frame, (640, 640))
            inputs = np.array([frame]).transpose((0, 3, 1, 2)).astype(float)
            inputs /= 255
            outputs = run_model(yolo, inputs)
            new_img, final_boxes, final_classes, final_confidences = draw_box(frame, outputs, True, car_label,
                                                                              False)  # 画框体

            if need_track or need_speed:
                if len(final_boxes) == 0:
                    continue
                tracked_boxes, new_boxes_ids = track_objects(tracker, new_img, final_boxes, final_confidences, mars_pb)
                if not need_speed:
                    new_img = draw_id_speed(new_img, tracked_boxes, new_boxes_ids)
                    new_img = draw_rec(new_img, tracked_boxes)
                else:
                    if pixel_speed_function == get_flow_pixel_speeds:
                        pixel_speeds = pixel_speed_function(tracked_boxes, new_boxes_ids, frame)
                    elif pixel_speed_function == get_boxes_center_pixel_speeds:
                        pixel_speeds = pixel_speed_function(tracked_boxes, new_boxes_ids, frame)
                        # TODO: 更多测速函数
                    new_img = draw_id_speed(new_img, tracked_boxes, new_boxes_ids, pixel_to_real_speed(pixel_speeds))
                    new_img = draw_rec(new_img, tracked_boxes)
            else:
                new_img = draw_rec(new_img, final_boxes)
            image_list.append(new_img)
            cv2.imshow('frame', new_img)
            cv2.setWindowTitle('frame', 'result')
            key = cv2.waitKey(delay=1)
            if key == ord("q"):
                break
        height, width, layers = image_list[0].shape
        fps = 30
        video = cv2.VideoWriter('./result.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
        for img in image_list:
            video.write(img)
        video.release()
    elif os.path.splitext(source)[1] in image_format:
        image = cv2.imread(source, cv2.IMREAD_COLOR)
        inputs = cv2.resize(image, (640, 640)).astype(float)
        inputs = np.array([inputs]).transpose((0, 3, 1, 2))
        inputs /= 255
        outputs = run_model(yolo, inputs)
        new_img, final_boxes, final_classes, final_confidences = draw_box(image, outputs, True, car_label, True)
        cv2.imshow('frame', new_img)
        cv2.setWindowTitle('frame', 'result')
        cv2.waitKey(0)
    else:
        print(f"error format {os.path.splitext(source)[1]}")


if __name__ == "__main__":
    # pass
    # run_plate(if_yoloPlate_pt, lpr_pt, "./test_data/OIP.jpg", 0)
    run_car(sa_yoloCar_pt, "./test_data/test.mp4", 6, True, get_boxes_center_pixel_speeds, True)
