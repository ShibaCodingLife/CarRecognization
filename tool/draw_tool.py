import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tool.process import get_all_box, non_maximum_suppression


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


def draw_id_speed(image, boxes, ids, real_speeds=None):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for i in range(len(ids)):
        x1, x2, y1, y2 = boxes[i]
        y = y1 - 5 if y1 > 5 else y1
        if real_speeds is not None:
            if real_speeds[i] is not None:
                pass
            else:
                real_speeds[i] = 0
            draw.text((x1, y), str(ids[i]) + ": " + str(1 + int(real_speeds[i])), font=font2, fill='rgb(0, 255, 0)')
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
