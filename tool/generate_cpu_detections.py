# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse

import cv2
# from openvino.inference_engine import IECore, ExecutableNetwork
import openvino.runtime as ov
import numpy as np
from openvino.inference_engine import IECore
from openvino.runtime import CompiledModel


def build_ov_model(xml: str, bin: str):
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
    model.reshape({input_layer_name: [-1, 128, 64, 3]})
    exec_model = core.compile_model(model, 'CPU')
    return exec_model, input_layer_name, output_layer_name


def run_ov_model(model: CompiledModel, input_data: np.ndarray, input_layer_name, output_layer_name):
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


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


class ovImageEncoder(object):

    def __init__(self, mars_xml, mars_bin):
        self.model, self.input_name, self.output_name = build_ov_model(mars_xml, mars_bin)
        self.image_shape = [64, 128]

    def __call__(self, data_x):
        out = run_ov_model(self.model, data_x, self.input_name, self.output_name)
        return out


image_encoder = None


def create_ov_box_encoder(model_xml, model_bin):  # OpenVINO
    global image_encoder
    if image_encoder is None:
        image_encoder = ovImageEncoder(model_xml, model_bin)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        features = []
        for box in boxes:
            x1, x2, y1, y2 = box
            patch = image[int(y1):int(y2), int(x1):int(x2), :]
            patch = cv2.resize(patch, image_shape)
            feature = image_encoder(np.array([patch])).squeeze(0)
            features.append(feature)
        return np.array(features)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
                                "standard MOT detections Directory structure should be the default "
                                "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
                             " exist.", default="detections")
    return parser.parse_args()
