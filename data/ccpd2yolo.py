"""
调整CCPD的路径结构，并划分数据集
原数据集路径结构：
CCPD----------------
                |----------CCPD
                                |--------ccpd_base(CCPD2019中的文件夹之一)
                                |--------ccpd_blur(CCPD2019中的文件夹之一)
                                |--------……
                                |--------ccpd_green(CCPD2020中的图像，把tain/test/val文件夹中的文件全放到ccpd_green中，然后删除train/test/val).如果需要将绿色车牌和蓝色车牌混在一起训练，就可以加上这个文件夹

首先进行数据划分，重新划分数据后，目录结构为：
CCPD----------------
                |-----------CCPD
                                |-----------train
                                |-----------test
                                |-----------val

然后转换为COCO类似的组织形式，并生成yaml文件最终转换为COCO的目录结构：
CCPD--------------
            |-----------CCPD
                            |------------train
                                            |------------labels
                                                            |------------train
                                                            |------------val
                                            |------------images
                                                            |------------train
                                                            |------------val
                            |------------test
"""
import os
import os.path
import shutil
import random
from tqdm import tqdm
import cv2
import yaml


def mv_img_in_dir(source_dir: str, target_dir: str):
    is_img_file = [".jpg", ".bmp", "jpeg", ".png"]
    if not os.path.exists(source_dir):
        return None
    files = os.listdir(source_dir)
    for file_name in tqdm(files, desc="正在移动文件", unit='file', ncols=80):
        if os.path.isfile(file_name) and os.path.splitext(file_name)[1] in is_img_file:
            source_file_path = os.path.join(source_dir, file_name)
            target_file_path = os.path.join(target_dir, file_name)
            if os.path.exists(source_file_path):
                shutil.move(source_file_path, target_file_path)


def mv_dir_by_ratio(source_dir: str, target_root_dir: str):
    """
    将原目录的内容按照比例7:2:1，全部移动到目标目录的相应子目录，然后删除原目录
    :param source_dir:原目录（ccpd_base/ccpd_blur……）
    :param target_root_dir:train/test/val 目录所在的目录
    :return:
    """
    # 获取划分好的list
    file_paths = []
    for file_name in tqdm(os.listdir(source_dir), desc='获取划分前的数据', ncols=80, unit='file'):
        file_path = os.path.join(source_dir, file_name)
        file_paths.append(file_path)
    print("正在划分数据集")
    train_list, val_list, test_list = random_split_list(file_paths)
    del file_paths
    print("划分完成")
    print("正在检查目录结构")
    # 检查train/test/val文件夹
    train_dir = os.path.join(target_root_dir, 'train')
    test_dir = os.path.join(target_root_dir, 'test')
    val_dir = os.path.join(target_root_dir, 'val')
    if (not os.path.exists(train_dir)) or (not os.path.isdir(train_dir)):
        os.mkdir(train_dir)
    if (not os.path.exists(test_dir)) or (not os.path.isdir(test_dir)):
        os.mkdir(test_dir)
    if (not os.path.exists(val_dir)) or (not os.path.isdir(val_dir)):
        os.mkdir(val_dir)
    print("检查完成")
    # 移动文件
    print("正在移动文件")
    for img in tqdm(train_list, desc='移动数据', ncols=80, unit='file'):
        image_name = os.path.basename(img)
        target_path = os.path.join(train_dir, image_name)
        shutil.move(img, target_path)
    del train_list
    for img in tqdm(test_list, desc='移动数据', ncols=80, unit='file'):
        image_name = os.path.basename(img)
        target_path = os.path.join(test_dir, image_name)
        shutil.move(img, target_path)
    del test_list
    for img in tqdm(val_list, desc='移动数据', ncols=80, unit='file'):
        image_name = os.path.basename(img)
        target_path = os.path.join(val_dir, image_name)
        shutil.move(img, target_path)
    del val_list
    print("移动完成")
    os.rmdir(source_dir)
    del source_dir


def random_split_list(original_list: list):  # 采用的是随机打乱后进行切分的方法。随机抽样的方法比较慢
    """
    划分数据集
    :param original_list:划分前的list
    :return: 划分后的list的tuple
    """
    n = len(original_list)
    # 计算每个部分的长度
    part1_len = int(0.7 * n)
    part2_len = int(0.2 * n)
    # 随机打乱原始列表
    random.shuffle(original_list)  # 时间复杂度O（n*logn）

    # 划分列表
    part1 = original_list[:part1_len]
    part2 = original_list[part1_len:part1_len + part2_len]
    part3 = original_list[part1_len + part2_len:]

    return part1, part2, part3


def re_split_data(CCPD_path):
    for ccpd_dir in tqdm(os.listdir(CCPD_path), desc=f'尝试提取文件', ncols=80, unit='dir'):
        if ccpd_dir == 'train' or ccpd_dir == 'test' or ccpd_dir == 'val' or not os.path.isdir(
                os.path.join(CCPD_path, ccpd_dir)):
            continue
        if ccpd_dir != 'splits':
            ccpd_path = os.path.join(CCPD_path, ccpd_dir)
            mv_dir_by_ratio(ccpd_path, CCPD_path)
        else:
            ccpd_path = os.path.join(CCPD_path, ccpd_dir)
            shutil.rmtree(ccpd_path)  # 删除splits，这个文件夹中的内容没用了


def get_box_pos(file_path):
    basename = os.path.basename(file_path)
    try:
        _, _, box_pos, points, plate_num, brightness, _ = basename.split('-')
    except ValueError:
        basename = os.path.basename(file_path)
        dirname = os.path.dirname(file_path)  # train/val
        dirname = os.path.dirname(dirname)  # images
        dirname = os.path.dirname(dirname)  # train
        dirname = os.path.dirname(dirname)  # CCPD
        new_path = os.path.join(dirname, basename)
        print(
            f"该文件{file_path}命名格式有误，已自动为您将该文件移动到{dirname}目录下，如果该图像是车牌图像，您可以手动将其移动到测试文件夹test中")
        shutil.move(file_path, new_path)
        return None
    upper_left, bottom_right = box_pos.split("_", 1)  # 第二次分割，以下划线'_'做分割
    xmin, ymin = upper_left.split("&", 1)
    xmax, ymax = bottom_right.split("&", 1)
    # 获取宽和高
    width = int(xmax) - int(xmin)
    height = int(ymax) - int(ymin)
    # 获取中心点
    x_center = float(xmin) + width / 2
    y_center = float(ymin) + height / 2
    # 进行归一化
    image = cv2.imread(file_path)
    if image is None:
        return None
    width = width / image.shape[1]
    height = height / image.shape[0]
    x_center = x_center / image.shape[1]
    y_center = y_center / image.shape[0]
    return x_center, y_center, width, height


def generate_labeltxt(image_dir: str, label_dir: str):
    for image_name in tqdm(os.listdir(image_dir), desc=f"正在获取label", unit="file", ncols=80):
        txt_name = image_name.split(".", 1)[0]
        txt_name += '.txt'
        txt_path = os.path.join(label_dir, txt_name)
        if os.path.exists(txt_path):
            continue
        # 获取box
        image_path = os.path.join(image_dir, image_name)
        box_pos = get_box_pos(image_path)
        if box_pos is None:
            continue
        label_str = str(0) + " " + str(box_pos[0]) + " " + str(box_pos[1]) + " " + str(box_pos[2]) + " " + str(
            box_pos[3])
        with open(txt_path, 'w') as f:
            f.write(label_str)


def generate_yaml(message,yamlpath):
    print("正在生成yaml文件")
    with open(yamlpath, 'w') as file:
        yaml.dump(message, file)
    print(f"已生成yaml文件，路径：{yamlpath}")


def convert_to_coco_like(root_path):
    """
    将重新划分后的数据集转换为和coco类似的组织形似，然后生成yaml文件
    :return:
    """
    # 生成相应的目录
    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'val')
    train_images_path = os.path.join(train_path, 'images')
    train_labels_path = os.path.join(train_path, 'labels')
    train_images_train_path = os.path.join(train_images_path, 'train')
    train_images_val_path = os.path.join(train_images_path, 'val')
    train_labels_train_path = os.path.join(train_labels_path, 'train')
    train_labels_val_path = os.path.join(train_labels_path, 'val')
    for path in (
            train_images_path, train_labels_path, train_images_train_path, train_images_val_path,
            train_labels_train_path,
            train_labels_val_path):
        if not os.path.exists(path):
            os.mkdir(path)
    # 移动数据
    mv_img_in_dir(val_path, train_images_val_path)
    mv_img_in_dir(train_path, train_images_train_path)
    if os.path.exists(val_path):
        os.rmdir(val_path)
    # 生成label并写入txt文件，然后放在labels中
    generate_labeltxt(train_images_train_path,train_labels_train_path)
    generate_labeltxt(train_images_val_path,train_labels_val_path)
    # 生成yaml文件
    message = {
        'path': root_path,
        'train': 'train/images/train',
        'val': 'train/images/val',
        'test': 'test',
        'nc': 1,
        'names': ['license']
    }
    yaml_path = os.path.join(root_path, 'CCPD.yaml')
    generate_yaml(message,yaml_path)

def CCPD2yolo(root_path):  # C:\CCPD\CCPD
    CCPD_path = root_path
    re_split_data(CCPD_path=CCPD_path)
    convert_to_coco_like(root_path=CCPD_path)


if __name__ == "__main__":
    CCPD2yolo(r'C:\CCPD\CCPD')
