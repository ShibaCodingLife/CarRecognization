import torch
import torch.nn as nn
import numpy as np


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),  # 降维，改变通道数
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),  # 宽卷积，捕捉竖直方向上的特征，通过padding保证形状不变
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),  # 宽卷积，捕捉水平方向上的特征，通过padding保证形状不变
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),  # 升维，改变通道数
        )

    def forward(self, x):
        return self.block(x)


class MaxPool3d_muti_batch(torch.nn.Module):
    # for multi one batch
    def __init__(self, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)):
        super(MaxPool3d_muti_batch, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_pool_2d = torch.nn.MaxPool2d(kernel_size[1:], self.stride[1:], padding[1:])
        self.max_pool_1d = torch.nn.MaxPool1d(kernel_size=kernel_size[0], stride=self.stride[0],
                                              padding=self.padding[0])  # stride is kernal_size

    def forward(self, x):
        batch = x.size()[0]
        res = []
        for i in range(batch):
            x_temp = x[i].unsqueeze(0)
            x_temp = self.max_pool_2d(x_temp)
            x_temp = x_temp.squeeze(0).permute(1, 2, 0)
            x_temp = self.max_pool_1d(x_temp)
            x_temp = x_temp.permute(2, 0, 1).unsqueeze(0)
            res.append(x_temp)
        return torch.cat(res, dim=0)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0  [bs,3,24,94] -> [bs,64,22,92]
            nn.BatchNorm2d(num_features=64),  # 1  -> [bs,64,22,92]
            nn.ReLU(),  # 2  -> [bs,64,22,92]
            MaxPool3d_muti_batch(kernel_size=(1, 3, 3), stride=(1, 1, 1)),  # 3  -> [bs,64,20,90]

            small_basic_block(ch_in=64, ch_out=128),  # 4  -> [bs,128,20,90]
            nn.BatchNorm2d(num_features=128),  # 5  -> [bs,128,20,90]
            nn.ReLU(),  # 6  -> [bs,128,20,90]
            MaxPool3d_muti_batch(kernel_size=(1, 3, 3), stride=(2, 1, 2)),  # 7  -> [bs,64,18,44]

            small_basic_block(ch_in=64, ch_out=256),  # 8  -> [bs,256,18,44]
            nn.BatchNorm2d(num_features=256),  # 9  -> [bs,256,18,44]
            nn.ReLU(),  # 10 -> [bs,256,18,44]

            small_basic_block(ch_in=256, ch_out=256),  # 11 -> [bs,256,18,44]
            nn.BatchNorm2d(num_features=256),  # 12 -> [bs,256,18,44]
            nn.ReLU(),  # 13 -> [bs,256,18,44]
            MaxPool3d_muti_batch(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14 -> [bs,64,16,21]

            nn.Dropout(dropout_rate),  # 0.5 dropout rate                          # 15 -> [bs,64,16,21]

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16 -> [bs,256,16,18]
            nn.BatchNorm2d(num_features=256),  # 17 -> [bs,256,16,18]
            nn.ReLU(),  # 18 -> [bs,256,16,18]

            nn.Dropout(dropout_rate),  # 0.5 dropout rate                                  19 -> [bs,256,16,18]

            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            # class_num=68  20  -> [bs,68,4,18]
            nn.BatchNorm2d(num_features=class_num),  # 21 -> [bs,68,4,18]
            nn.ReLU(),  # 22 -> [bs,68,4,18]
        )

        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )
        # self.connected = nn.Sequential(
        #     nn.Linear(class_num * 88, 128),
        #     nn.ReLU(),
        # )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)  # 保存第2/6/13/22层的输出

        global_context = list()
        # keep_features: [bs,64,22,92]  [bs,128,20,90] [bs,256,18,44] [bs,68,4,18]
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                # [bs,64,22,92] -> [bs,64,4,18]
                # [bs,128,20,90] -> [bs,128,4,18]
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                # [bs,256,18,44] -> [bs,256,4,18]
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            # *            # 进行归一化
            f_pow = torch.pow(f, 2)  # [bs,64,4,18]  所有元素求平方
            f_mean = torch.mean(f_pow)  # 1 所有元素求平均
            f = torch.div(f, f_mean)  # [bs,64,4,18]  所有元素除以这个均值
            global_context.append(f)

        # 拼接
        x = torch.cat(global_context, 1)  # [bs,516,4,18]
        x = self.container(x)  # -> [batch_size, 68, 4, 18]   head头
        logits = torch.mean(x, dim=2)  # -> [batch_size, 68, 18]  # 68 字符类别数   18字符序列长度

        return logits

    def predict(self, data: np.ndarray):
        image = torch.tensor([data, ], dtype=torch.float32)
        image = torch.transpose(image, dim0=2, dim1=0)
        image = torch.transpose(image, dim0=1, dim1=2)
        ans = self.forward(image)
        return ans

        # https://blog.csdn.net/weixin_39027619/article/details/106143755
    # def forward(self, x):
    #     x = self.backbone(x)
    #     pattern = x.flatten(1, -1)
    #     pattern = self.connected(pattern)
    #     width = x.size()[-1]
    #     pattern = torch.reshape(pattern, [-1, 128, 1, 1])
    #     pattern = pattern.repeat(1, 1, 1, width)
    #     x = torch.cat([x, pattern], dim=1)
    #     x = self.container(x)
    #     logits = x.squeeze(2)
    #     return logits


def build_lprnet(lpr_max_len=8, class_num=68, dropout_rate=0.5, phase='train'):
    Net = LPRNet(lpr_max_len, class_num, dropout_rate)
    if phase == "train":
        return Net.train()
    else:
        return Net.eval()