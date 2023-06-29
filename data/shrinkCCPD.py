"""
如果CCPD数据集过大，可以运行该代码来生成一份缩小版的CCPD副本

代码首先会确保将路径的组织形式转换成以下格式
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

然后代码在CCPD下生成shrinkCCPD目录用来存放缩小版的CCPD副本
CCPD------------
            |------------CCPD
                                        |------------train
                                            |------------labels
                                                            |------------train
                                                            |------------val
                                            |------------images
                                                            |------------train
                                                            |------------val
                            |------------test
            |------------shrinkCCPD
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
from tqdm import tqdm
import random


def check_and_clear_directory(directory):
    if len(os.listdir(directory)) != 0:
        while True:
            user_input = input(f"{directory}目录非空，是否清空该目录？ (y/n): ")
            if user_input.lower() == "y":
                for file_name in tqdm(os.listdir(directory), desc='正在清空目录', unit='file', ncols=100):
                    file_path = os.path.join(directory, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print("目录已清空")
                break
            elif user_input.lower() == "n":
                print("操作已取消，即将退出")
                exit(0)
            else:
                print("无效的输入，请重新输入")


def copy_directory_structure(src_dir, dest_dir):
    """
    将源目录的目录结构拷贝到目标目录，只生成相应的文件夹
    :param src_dir: 源目录
    :param dest_dir: 目标目录
    :return:
    """
    for root, dirs, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        new_dir = os.path.join(dest_dir, relative_path)
        # 创建目录
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)


def copy_files_to_directory(file_list, dest_dir):
    """
    将列表中的文件全部拷贝到指定文件夹
    :param file_list: 文件列表
    :param dest_dir: 目标文件夹
    :return:
    """
    for file_path in tqdm(file_list, desc='正在生成副本', ncols=100, unit='file'):
        file_name = os.path.basename(file_path)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.copy2(file_path, dest_file)


def shrinkCCPD(CCPDpath, shrinkCCPDpath, shrink_rate=0.4, convert_dir=True):
    """
    生成一份缩小版的CCPD副本
    :param convert_dir: 是否要对CCPD数据集的文件夹进行转换（可以重复转换）
    :param CCPDpath: CCPD的位置。例如：./CCPD/CCPD
    :param shrinkCCPDpath: shrinkCCPD的位置。例如：./CCPD/shrinkCCPD
    :param shrink_rate: shrink后的CCPD图像数量和原CCPD图像数量的比值
    :return:
    """
    if convert_dir:
        from data.ccpd2yolo import CCPD2yolo
        CCPD2yolo(CCPDpath)
    ori_train_images_path = os.path.join(CCPDpath, 'train', 'images', 'train')
    ori_val_images_path = os.path.join(CCPDpath, 'train', 'images', 'val')
    ori_test_images_path = os.path.join(CCPDpath, 'test')
    new_train_images_path = os.path.join(shrinkCCPDpath, 'train', 'images', 'train')
    new_val_images_path = os.path.join(shrinkCCPDpath, 'train', 'images', 'val')
    new_test_images_path = os.path.join(shrinkCCPDpath, 'test')
    new_train_labels_path = os.path.join(shrinkCCPDpath, 'train', 'labels', 'train')
    new_val_labels_path = os.path.join(shrinkCCPDpath, 'train', 'labels', 'val')
    new_yaml_path = os.path.join(shrinkCCPDpath, 'shrinkCCPD.yaml')
    # 生成目录结构
    print("正在检查并生成目录")
    if not os.path.exists(shrinkCCPDpath):
        os.makedirs(shrinkCCPDpath)
    copy_directory_structure(CCPDpath, shrinkCCPDpath)
    for new_dir in (new_test_images_path, new_train_images_path, new_val_images_path, new_train_labels_path,
                    new_val_labels_path):
        check_and_clear_directory(new_dir)
    # 对数据进行抽样
    sam_train_imglist = []
    sam_val_imglist = []
    sam_test_imglist = []
    for images_dir in tqdm((ori_train_images_path, ori_val_images_path, ori_test_images_path), desc='抽样',
                           ncols=100, unit='dir'):
        imglist = []
        for image_name in tqdm(os.listdir(images_dir), desc='正在获取原文件列表', ncols=100, unit='file'):
            image_path = os.path.join(images_dir, image_name)
            imglist.append(image_path)
        print(f"正在按{shrink_rate}比例进行抽样")
        sampled_list = random.sample(imglist, int(len(imglist) * shrink_rate))
        if images_dir == ori_train_images_path:
            sam_train_imglist = sampled_list
        elif images_dir == ori_val_images_path:
            sam_val_imglist = sampled_list
        else:
            sam_test_imglist = sampled_list
    # 拷贝文件
    copy_files_to_directory(sam_train_imglist, new_train_images_path)
    copy_files_to_directory(sam_test_imglist, new_test_images_path)
    copy_files_to_directory(sam_val_imglist, new_val_images_path)
    # 生成labels
    from ccpd2yolo import generate_labeltxt
    from ccpd2yolo import generate_yaml
    generate_labeltxt(new_train_images_path, new_train_labels_path)
    generate_labeltxt(new_val_images_path, new_val_labels_path)
    # 生成yaml
    message = {
        'path': shrinkCCPDpath,
        'train': 'train/images/train',
        'val': 'train/images/val',
        'test': 'test',
        'nc': 1,
        'names': ['license']
    }
    generate_yaml(message, new_yaml_path)


if __name__ == "__main__":
    CCPD=r"C:\CCPD\CCPD"
    shrink=r"C:\CCPD\shrinkCCPD"
    shrinkCCPD(CCPD,shrink,0.4,False)
