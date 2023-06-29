# -*- coding: utf-8 -*-
"""
提供了用于训练LPRNet的Dataset,初始化Dataset时会运行调整路径结构的函数（该函数可以多次重复运行）
"""
import os
import os.path
import cv2
import torch
from torch.utils.data import Dataset


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂",
             "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂',
         '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '0', '1', '2', '3', '4', '5',
         '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'W', 'X', 'Y', 'Z', 'I', 'O', '-'] # 68
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
ONEHOT_DICT = {i: char for i, char in enumerate(CHARS)}


def get_plate(file_path):
    """
    作用：根据文件名获取车牌号和车牌位置
    :param file_path: file_path:文件路径
    :return: 车牌号字符串
    """
    basename = os.path.basename(file_path)
    file_name = os.path.splitext(basename)[0]
    _, _, box_pos, points, plate_num, brightness, blurriness = file_name.split('-')
    list_plate = plate_num.split('_')  # 读取车牌，但是CCPD的车牌号需要进行转换菜式真正的车牌号
    # 获取车牌号
    number = ""  # 保存车牌号
    number += provinces[int(list_plate[0])]
    number += alphabets[int(list_plate[1])]
    number += ads[int(list_plate[2])] + ads[int(list_plate[3])] + ads[int(list_plate[4])] + ads[int(list_plate[5])] + \
              ads[int(list_plate[6])]
    if len(list_plate) == 8:
        number += ads[int(list_plate[7])]
    # 获取车牌位置
    box_pos = box_pos.split('_')  # 车牌边界
    box_pos = [list(map(int, i.split('&'))) for i in box_pos]
    xmin = box_pos[0][0]
    xmax = box_pos[1][0]
    ymin = box_pos[0][1]
    ymax = box_pos[1][1]
    pos = (xmin, xmax, ymin, ymax)
    return pos, number


def get_from_yolo_CCPD(root_dir, name):
    """
    作用：从CCPD中获取图像文件路径与相应的车牌信息
        文件组织形式：CCPD / CCPD / train+test
    :param root_dir: root_dir是CCPD2020数据集根目录，root_dir / CCPD / name
    :param name: train，test或者val
    :return: 图像路径列表，车牌信息列表
    """
    if name == 'test':
        image_dir = os.path.join(root_dir,name)
    else:
        image_dir = os.path.join(root_dir, 'train', 'images', name)
    image_path_list = []
    plate_pos_list = []
    plate_number_list = []
    for path in os.listdir(image_dir):
        # 获取图像路径
        image_path = os.path.join(image_dir, path)
        image_path_list.append(image_path)
        # 获取车牌信息
        plate_pos, plate_number = get_plate(image_path)
        plate_pos_list.append(plate_pos)
        plate_number_list.append(plate_number)
    return image_path_list, plate_pos_list, plate_number_list


def plate2onehot(plateStr, maxLen=8):
    """
    将车牌str转为one-hot并统一长度。统一长度的作用是为了方便batch操作
    :param plateStr: 车牌号string
    :param maxLen: 车牌号的最大长度，默认为8，小于此长度的车牌号将会被用‘-’填充为此长度。
    :return:
    """
    plateLen = len(plateStr)
    onehot = []
    for c in plateStr:
        onehot.append(CHARS_DICT[c])
    while plateLen < maxLen:
        onehot.append(CHARS_DICT['-'])
        plateLen += 1
    return torch.tensor(onehot, dtype=torch.long)


def onehot2plate(onehot):
    """
    将onehot转为车牌号
    :param onehot: 一热编码
    :return: 车牌号string
    """
    plateStr = ""
    for num in onehot:
        plateStr += ONEHOT_DICT[num]
    return plateStr


def imageProcessing(image):
    """
    进行归一化并将opencv读取到的image转置为pytorch的CNN输入格式
    :param image: opencv读取到的image对象
    :return: 处理后的image，类型为torch.tensor ,dtype=float32
    """
    image = torch.tensor(image, dtype=torch.float32)
    image -= 127.5
    image /= 127.5
    image = torch.transpose(image, dim0=2, dim1=0)
    image = torch.transpose(image, dim0=1, dim1=2)
    return image


class licenseDataset(Dataset):
    """
    pytorch形式的用于车牌识别的CCPD车牌数据集类
    """

    def __init__(self, root_dir, parse_list,convert_dir=True):
        """
        初始化函数
        :param root_dir: CCPD的位置。例如：./CCPD/CCPD
        :param parse_list: 可以用for迭代的对象，其中item的取值为train / test / val
        """
        if convert_dir:
            from data.ccpd2yolo import CCPD2yolo
            CCPD2yolo(root_dir)
        self.image_path_list = []
        self.plate_pos_list = []
        self.plate_number_list = []
        parse_exist = []
        for parse in parse_list:
            if not isinstance(parse, str):
                print(f"类型错误，parse_list中的元素必须是python字符串类型，parse_list可以是字符串列表")
            if parse in parse_exist:
                print(f"重复指定{parse}数据集，已自动跳过")
                continue
            else:
                parse_exist.append(parse)
            if parse == 'train' or parse == 'test' or parse == 'val':
                im, pp, pn = get_from_yolo_CCPD(root_dir, parse)
                self.image_path_list += im
                self.plate_pos_list += pp
                self.plate_number_list += pn

    def __getitem__(self, index):
        """
        根据索引获取数据集内容
        :param index: 索引
        :return: 图像，车牌onehot标签，车牌号长度，车牌号string。车牌号长度用于指定CTCloss的target_length
        """
        image_path = self.image_path_list[index]
        xmin, xmax, ymin, ymax = self.plate_pos_list[index]
        plateStr = self.plate_number_list[index]
        image = cv2.imread(image_path)
        # 只保留车牌
        image = image[ymin:ymax, xmin:xmax]
        image = cv2.resize(image, (94, 24))  # 24*94*3
        # cv2.imshow('',image)
        # cv2.waitKey(0)

        # 对image进行预处理
        image = imageProcessing(image)

        # 根据CHARS将plateStr从长度不一的str类型转为长度为8的tensor
        onehotLabel = plate2onehot(plateStr)
        return image, onehotLabel, len(plateStr), plateStr

    def __len__(self):
        """
        :return:数据集长度
        """
        return len(self.image_path_list)


if __name__ == '__main__':
    datas = licenseDataset(r'C:\CCPD\CCPD', ['train'])
    image, onehot, plateLen, plate_number = datas.__getitem__(1)
    print(f'蓝色车牌，图像的shape：{image.shape, image.max(), image.min(), image.dtype}', '   ', f"车牌号：{plate_number}",
          '   ',
          f'车牌号长度：{plateLen}', "   ", f"onehot编码：{onehot}")
    image, onehot, plateLen, plate_number = datas.__getitem__(105760)
    print(f'绿色车牌，图像的shape：{image.shape}', '   ', f"车牌号：{plate_number}", '   ',
          f'车牌号长度：{plateLen}', "   ", f"onehot编码：{onehot}")
    print(f'数据集长度：{datas.__len__()}')
    print(CHARS_DICT)
    print(ONEHOT_DICT)
