import numpy as np
import cv2

from data.ccpd2lpr import CHARS


def get_lpr_input(image, boxes):
    data_list = []
    for box in boxes:
        x1, x2, y1, y2 = box
        lpr_area = image[int(y1):int(y2):, int(x1):int(x2):, :]
        lpr_area = cv2.resize(lpr_area, (94, 24))  # 24,94,3
        lpr_input = np.array(lpr_area).transpose((2, 0, 1)).astype(float)  # 3,24,94
        lpr_input /= 255  # 归一化
        data_list.append(lpr_input)
    return np.array(data_list, dtype=np.float)


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


def get_lpr_outstr(output: np.ndarray):  # output:[bs,68,18]
    """
    Args:
        output: LPRNet的输出
    Returns:str的list，一共batch_size个str
    """
    str_list = []
    print(output.shape)
    for i in range(output.shape[0]):
        preb = output[i]  # 68,18
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
