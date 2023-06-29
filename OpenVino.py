import os.path

import cv2
from PIL import Image, ImageDraw, ImageFont
# from openvino.inference_engine import IECore, ExecutableNetwork
from openvino.inference_engine import IECore
from openvino.runtime import CompiledModel

from data.ccpd2lpr import CHARS
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from tool.optical_flow import optical_tracker
from tool.track_tool import track_objects

# from tool.track_tool import track_objects

# from tool.track_tool import track_objects

image_format = [".jpg", ".jepg"]
video_format = [".mp4"]

lpr_xml = "./weights/LPRNet/OpenVINO/mybestLPRNet.xml"
lpr_bin = "./weights/LPRNet/OpenVINO/mybestLPRNet.bin"
if_yoloCar_xml = r"./weights/yoloCar/inaccurate_fast/OpenVINO/yolo_for_car.xml"
if_yoloCar_bin = r"./weights/yoloCar/inaccurate_fast/OpenVINO/yolo_for_car.bin"
if_yoloPlate_xml = r"./weights/yoloPlate/inaccurate_fast/OpenVINO/yolo_for_plate.xml"
if_yoloPlate_bin = r"./weights/yoloPlate/inaccurate_fast/OpenVINO/yolo_for_plate.bin"
sa_yoloCar_xml = r"./weights/yoloCar/slow_accurate/OpenVINO/best.xml"
sa_yoloCar_bin = r"./weights/yoloCar/slow_accurate/OpenVINO/best.bin"
sa_yoloPlate_xml = r"./weights/yoloPlate/slow_accurate/OpenVINO/yolo_for_plate.xml"
sa_yoloPlate_bin = r"./weights/yoloPlate/slow_accurate/OpenVINO/yolo_for_plate.bin"
fst_yoloCar_bin = r".\weights\yoloCar\fastest\OpenVINO\best.bin"
fst_yoloCar_xml = r".\weights\yoloCar\fastest\OpenVINO\best.xml"
mars_xml = r"./weights/mars/OpenVINO/mars.xml"
mars_bin = r"./weights/mars/OpenVINO/mars.bin"

test_Car_source = ""
test_plate_source = ""
metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)

# Initialize the tracker
tracker = Tracker(metric)

# Process multiple frames
# for frame, boxes in video:
#     tracked_boxes,ids = track_objects(tracker, frame, boxes)

import openvino.runtime as ov
import numpy as np


def build_model(xml: str, bin: str):
    """
    创建OpenVINO模型
    Args:
        xml: xml文件地址
        bin: bin文件地址
    Returns:可执行模型，输入层名字，输出层名字
    """
    core = ov.Core()
    ie = IECore()
    model = core.read_model(model=xml, weights=bin)
    model1 = ie.read_network(model=xml, weights=bin)
    input_layer_name = next(iter(model1.input_info))
    output_layer_name = next(iter(model1.outputs))

    exec_model = core.compile_model(model, device_name="CPU")
    return exec_model, input_layer_name, output_layer_name


def run_model(model: CompiledModel, input_data: np.ndarray, input_layer_name, output_layer_name):
    """
    运行模型
    Args:
        model: 可执行模型
        input_data: 输入的图像数据
        input_layer_name: 输入层名字
        output_layer_name: 输出层名字
    Returns:输出层的输出
    """
    if input_data.shape[0] == 1:
        return model(input_data)[output_layer_name]
    for input_img in input_data:
        outputs = []
        outputs.append(model(input_img)[output_layer_name])
    return outputs


def get_all_box(output: np.ndarray):
    """
    返回8400个框（1，4，8400）4分别是cx,cy,w,h，8400个框体类别判断的置信度（1，8400），8400个框体类别（1，8400）
    Args:output: 模型输出
    Returns:
    """
    # boxes = output.transpose((0, 2, 1))  # [batch size,8400,class num+4]
    # 对每个锚框的预测信息进行解码，以获取边界框的坐标、宽度、高度等信息
    boxes_info = output[:, :4:, ::]  # [batch size,4,8400]
    classes_info = output[:, 4:, ::]  # [batch size,class num,8400]
    confidence_info = np.max(classes_info, axis=1)  # 置信度[batch size,8400]
    class_info = np.argmax(classes_info, axis=1)  # 框体中的物体预测[batch size,8400]
    return boxes_info, confidence_info, class_info


def non_maximum_suppression(boxes, scores, score_threshold, nms_threshold, classes):  # 同时转成x1,x2,y1,y2
    """非极大值抑制
    Args:
    boxes: `np.array` shape [4, 8400].
    scores: `np.array` shape [8400].
    threshold: float 表示用于确定框是否重叠过多的阈值.
    """
    cx = boxes[0, :]  # [8400]
    cy = boxes[1, :]
    w = boxes[2, :]
    h = boxes[3, :]
    x1 = cx - w / 2  # [8400]
    x2 = cx + w / 2
    y1 = cy - h / 2
    y2 = cy + h / 2
    new_boxes = np.stack((x1, x2, y1, y2), axis=-1)  # [8400,4]
    indices = cv2.dnn.NMSBoxes(new_boxes.tolist(), scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    keep = [new_boxes[i] for i in indices]
    keep_class = [classes[i] for i in indices]
    keep_scores = [scores[i] for i in indices]
    return np.array(keep), np.array(keep_class), np.array(keep_scores)


def draw_box(img, output, isCar, label: int, draw=True):  # TODO: isCar为真时绘制文字Car ; isCar为假时绘制车牌号
    """
    根据输出，在img上绘制信息的函数
    Args:
        draw:
        img:
        output:
        isCar:
    Returns:
    """
    image = cv2.resize(img, (640, 640))
    boxes, scores, class_info = get_all_box(output)
    if boxes.shape[0] != 1:
        print(f"Error shape boxes {boxes.shape}")
        exit(-1)
    new_boxes, new_classes, confidences = non_maximum_suppression(boxes[0], scores[0], 0.4, 0.8, class_info[0])
    if isCar:  # 车辆检测
        final_boxes = new_boxes[new_classes == label]  # [N,4]
        final_classes = new_classes[new_classes == label]  # [N]
        final_confidences = confidences[new_classes == label]
        color = (0, 255, 0)
    else:  # 车牌检测
        final_boxes = new_boxes[new_classes == label]
        final_classes = new_classes[new_classes == label]
        final_confidences = confidences[new_classes == label]
        color = (0, 0, 255)
    if draw:
        for i in range(final_classes.shape[0]):
            x1, x2, y1, y2 = final_boxes[i]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    return image, final_boxes, final_classes, final_confidences


font = ImageFont.truetype('./STSONG.TTF', size=25, encoding="utf-8")
font2 = ImageFont.truetype('./STSONG.TTF', size=20, encoding="utf-8")


def draw_text(image, boxes, texts):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for i in range(len(boxes)):
        x1, x2, y1, y2 = boxes[i]
        draw.text((x1, y1), texts[i], font=font, fill='rgb(0, 255, 0)')
    image = np.array(pil_image)
    return image


def get_lpr_outstr(model, input_name, output_name, input: np.ndarray, boxes):  # input: 640,640,3 ,dtype=unit8
    """
    返回LPRNet识别的车牌字符串的list（可能有多个车牌），索引和boxes项对应
    Args:
        model: 可执行LPRNet
        input_name: 输入层
        output_name: 输出层
        input: 输入
        boxes: 车牌位置（x1,x2,y1,y2）格式
    Returns:

    """
    str_list = []
    for box in boxes:
        (x1, x2, y1, y2) = box
        lpr_area = input[int(y1):int(y2):, int(x1):int(x2):, :]
        lpr_area = cv2.resize(lpr_area, (94, 24))  # 24,94,3
        lpr_input = np.array([lpr_area]).transpose((0, 3, 1, 2)).astype(float)  # 1,3,24,94
        lpr_input /= 255  # 归一化
        output = run_model(model, lpr_input, input_name, output_name)  # 1,68,18
        preb = output[0]  # 68,18
        preb_label = list()
        for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:  # 记录重复字符
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # 去除重复字符和空白字符'-'
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        plate_str = ""
        for ind in no_repeat_blank_label:
            plate_str += CHARS[ind]
        str_list.append(plate_str)
        return str_list


def pixel_to_real_speed(pixel_speed,  # 目标的像素速度，单位像素/帧。
                        ppm=15,  # 像素每米
                        fps=30):  # 视频的帧率，单位帧/秒。

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
        speeds.append(speed)
    return speeds


def draw_id_speed(image, boxes, ids, speeds=None):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for i in range(len(ids)):
        x1, x2, y1, y2 = boxes[i]
        y = y1 - 5 if y1 > 5 else y1
        if speeds is not None:
            if speeds[i] is not None:
                speeds[i] = pixel_to_real_speed(speeds[i])
            else:
                speeds[i] = 0
            draw.text((x1, y), str(ids[i]) + ": " + str(1 + int(speeds[i])), font=font2, fill='rgb(0, 255, 0)')
        else:
            draw.text((x1, y), str(ids[i]) + ": ", font=font2, fill='rgb(0, 255, 0)')

    image = np.array(pil_image)
    return image


def draw_rec(image, boxes):
    color = (0, 255, 0)
    for i in range(len(boxes)):
        x1, x2, y1, y2 = boxes[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
    return image


def run_plate(yoloPlate_xml, yoloPlate_bin, lprxml, lprbin, source, plate_label):
    yoloplate_model, yoloplate_input_name, yoloplate_output_name = build_model(xml=yoloPlate_xml, bin=yoloPlate_bin)
    lpr_modl, lpr_input_name, lpr_output_name = build_model(xml=lprxml, bin=lprbin)
    if os.path.splitext(source)[1] in video_format or isinstance(source, int):
        cap = cv2.VideoCapture(source)
        while cap.isOpened():  # 检查是否成功初始化，否则就 使用函数 cap.open()
            # Capture frame-by-frame  逐帧从摄像头中读取图像
            ret, frame = cap.read()  # ret 返回一个布尔值 True/False
            frame = cv2.resize(frame, (640, 640))  # 640,640,3
            inputs = np.array([frame]).transpose((0, 3, 1, 2)).astype(float)  # 1,3,640,640
            inputs /= 255
            outputs = run_model(yoloplate_model, inputs, yoloplate_input_name, yoloplate_output_name)
            new_img, final_boxes, final_classes, final_confidences = draw_box(frame, outputs, False, plate_label)
            plate_str_list = get_lpr_outstr(lpr_modl, lpr_input_name, lpr_output_name, frame, final_boxes)
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
        outputs = run_model(yoloplate_model, inputs, yoloplate_input_name, yoloplate_output_name)
        new_img, final_boxes, final_classes, final_confidences = draw_box(image, outputs, False, plate_label, True)
        plate_str_list = get_lpr_outstr(lpr_modl, lpr_input_name, lpr_output_name, image, final_boxes)
        new_img = draw_text(new_img, final_boxes, plate_str_list)
        cv2.imshow('frame', new_img)
        cv2.setWindowTitle('frame', 'result')
        cv2.waitKey(0)
    else:
        print(f"error format {os.path.splitext(source)[1]}")


def run_car(yoloCar_xml, yoloCar_bin, source, car_label: int, need_speed: bool = True, pixel_speed_function=None,
            need_track: bool = True):  # TODO :将车速检测模块加入到run_car_detect
    exe_model, input_name, output_name = build_model(xml=yoloCar_xml, bin=yoloCar_bin)
    if os.path.splitext(source)[1] in video_format or isinstance(source, int):
        image_list = []
        cap = cv2.VideoCapture(source)
        if need_speed:
            need_track = True
        while cap.isOpened():  # 检查是否成功初始化，否则就 使用函数 cap.open()
            # Capture frame-by-frame  逐帧从摄像头中读取图像
            ret, frame = cap.read()  # ret 返回一个布尔值 True/False
            if not ret:
                break
            h, w = frame.shape[0], frame.shape[1]  # 记录原图像大小
            frame = cv2.resize(frame, (640, 640))
            inputs = np.array([frame]).transpose((0, 3, 1, 2)).astype(float)
            inputs /= 255
            outputs = run_model(exe_model, inputs, input_name, output_name)

            new_img, final_boxes, final_classes, final_confidences = draw_box(frame, outputs, True, car_label,
                                                                              False)  # 画框体

            if need_track or need_speed:
                tracked_boxes, new_boxes_ids = track_objects(tracker, new_img, final_boxes, final_confidences,
                                                             (mars_xml,
                                                              mars_bin))
                if not need_speed:
                    new_img = draw_id_speed(new_img, tracked_boxes, new_boxes_ids)
                    new_img = draw_rec(new_img, tracked_boxes)
                else:
                    if pixel_speed_function == get_flow_pixel_speeds:
                        pixel_speeds = pixel_speed_function(tracked_boxes, new_boxes_ids, frame)
                    new_img = draw_id_speed(new_img, tracked_boxes, new_boxes_ids, pixel_speeds)
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
        outputs = run_model(exe_model, inputs, input_name, output_name)
        new_img, final_boxes, final_classes, final_confidences = draw_box(image, outputs, True, car_label, True)
        cv2.imshow('frame', new_img)
        cv2.setWindowTitle('frame', 'result')
        cv2.waitKey(0)
    else:
        print(f"error format {os.path.splitext(source)[1]}")


if __name__ == "__main__":
    run_plate(sa_yoloPlate_xml, sa_yoloPlate_bin, lpr_xml, lpr_bin, "./test_data/test.mp4", 0)
    run_car(sa_yoloCar_xml, sa_yoloCar_bin, "./test_data/test.mp4", 6, True, get_flow_pixel_speeds, True)
