import os.path

import onnx
import torch
import numpy

from ultralytics import YOLO
from LPRNet.LPRNet import LPRNet, build_lprnet


def get_yolo(model: str):
    if os.path.splitext(model)[1] in [".pt", ".pth", ".pkl"]:
        yolo = YOLO(model, task="detect")
        return yolo.model
    elif os.path.splitext(model)[1] == ".onnx":
        model = onnx.load(model)
        return model
    print(f"Error,invalid format{os.path.splitext(model)[1]},expect .pt .pth .pkl .onnx")
    exit(-1)


def get_lpr(model: str):
    if os.path.splitext(model)[1] in [".pt", ".pth", ".pkl"]:
        lpr = build_lprnet(phase="test")
        lpr.load_state_dict(torch.load(model))
        return lpr
    elif os.path.splitext(model)[1] == ".onnx":
        lpr = onnx.load(model)
        return lpr
    print(f"Error,invalid format{os.path.splitext(model)[1]},expect .pt .pth .pkl .onnx")
    exit(-1)


if __name__ == "__main__":
    model = get_yolo(
        r"C:\codes\python_codes\dl_practice\CarRecognization\weights\yoloCar\inaccurate_fast\yolo_for_car.onnx")
    print(model)
    print(type(model))
