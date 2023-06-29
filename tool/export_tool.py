import os
import os.path

import torch.onnx
import onnx

from LPRNet.LPRNet import LPRNet, build_lprnet
from ultralytics import YOLO


def yolo2export(ptPath: str, format, device: str = 'cpu'):
    yolo = YOLO(ptPath)
    yolo.model = yolo.model.to(torch.device(device))
    yolo.export(format=format, batch=1)


def lpr2onnx(ptPath: str, device: str = 'cpu', dynamic_axes: dict = None):
    device = torch.device(device)
    model_name = os.path.splitext(ptPath)[0]
    dummy_input = torch.rand(1, 3, 24, 94).to(device)
    model = build_lprnet(phase="val")
    model.load_state_dict(torch.load(ptPath))
    model.eval().to(device)
    output_onnx = '{}.onnx'.format(model_name)
    print("==> Exporting model to onnx format at'{}'".format(output_onnx))
    input_names = ["input"]
    output_names = ["output"]
    model = model
    # dynamic_axes 构建动态batch_size
    if dynamic_axes is not None:
        torch.onnx.export(model, dummy_input, output_onnx, verbose=True, input_names=input_names,
                          output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes
                          )
    else:
        torch.onnx.export(model, dummy_input, output_onnx, verbose=True, input_names=input_names,
                          output_names=output_names, opset_version=11
                          )
    print("==> Loading and checking exported model from '{}'".format(output_onnx))
    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Passed")


def onnx2openvino(onnxPath: str, input_name: list, output_name: list, input_shape: list):
    import openvino.runtime as ov
    from openvino.tools import mo

    f_onnx = onnxPath
    f_ov = os.path.splitext(f_onnx)[0] + ".xml"
    ov_model = mo.convert_model(
        input_model=onnxPath,
        model_name=r"../weights",
        framework='onnx',
        output=output_name,
        input=input_name,
        input_shape=input_shape,

        compress_to_fp16=True)  # export
    # Serialize the model with dynamic batch size
    ov.serialize(ov_model, f_ov)  # save


def onnx2tensorrt(onnx_file_path):
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in parser.get_error_iterator():
                print(error)
            return None
    print('Completed parsing of ONNX file')

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        dims = network.get_input(i).shape
        min_dims = dims
        min_dims[0] = 1
        max_dims = dims
        max_dims[0] = 100
        opt_dims = dims
        opt_dims[0] = 10
        profile.set_shape(network.get_input(i).name, min_dims, opt_dims, max_dims)
    config.add_optimization_profile(profile)

    print('Building an engine...')
    engine = builder.build_engine(network, config)
    if engine is None:
        print('Failed to build the engine.')
        return None
    print("Completed creating Engine")

    with open("model.engine", "wb") as f:
        f.write(engine.serialize())
    return engine


def torch_to_trt_dynamic(ptpath, device, input_shape=(1, 3, 24, 94), min_batch=1, max_batch=50):
    import torch
    import tensorrt as trt
    from torch2trt import torch2trt
    # 创建一个假的输入tensor用于模型转换
    x = torch.ones(input_shape, dtype=torch.float32).to(torch.device(device))
    model = build_lprnet(phase="val")
    model.load_state_dict(torch.load(ptpath))
    model.to(torch.device(device))
    # 使用torch2trt进行模型转换
    model_trt = torch2trt(model, [x])

    # 创建一个TensorRT builder和network
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 将模型转换为动态维度
    for i in range(network.num_inputs):
        dims = network.get_input(i).shape
        dims[0] = -1  # 将第一维设置为-1，表示动态维度
        network.get_input(i).shape = dims

    # 创建一个TensorRT优化配置文件
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30

    # 创建一个TensorRT优化profile，设置动态批量的范围
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, (min_batch,) + input_shape, (min_batch,) + input_shape,
                      (max_batch,) + input_shape)
    config.add_optimization_profile(profile)

    # 使用builder和config创建一个新的TensorRT engine
    engine = builder.build_engine(network, config)

    # 保存engine
    with open("model.trt", "wb") as f:
        f.write(engine.serialize())

    return engine


dynamic_axes = {  # 指定动态维度
    "input": {0: "batch_size"},  # 第0维是动态的，名字是"batch_size"
    "output": {0: "batch_size"},  # 第0维是动态的，名字是"batch_size"
}
if __name__ == "__main__":
    # lpr2onnx(r"C:\codes\python_codes\dl_practice\Car\weights\LPRNet\mybestLPRNet.pt", "cuda", dynamic_axes)
    engine = torch_to_trt_dynamic(r"C:\codes\python_codes\dl_practice\Car\weights\LPRNet\mybestLPRNet.pt",
                                  device="cuda")
    print(engine)
