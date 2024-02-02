import torch
import torch.nn as nn
import torch.nn.functional as F

time_series_length = 4096


class wdcnn_Net(nn.Module):

    def __init__(self, num_classes=28, input_channels=4):
        super(wdcnn_Net, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=64, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=4, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (time_series_length // 64), 128)  # 调整全连接层输入维度
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)  # 扁平化
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 小波变换这里如何使用？？？
import pywt


# import numpy as np


def wavelet_transform(data, wavelet='db5', level=1):
    """
    对输入数据进行小波变换。
    :param data: 输入数据，形状为(batch_size, channels, length)
    :param wavelet: 小波变换使用的小波母函数名称
    :param level: 小波变换的层级
    :return: 小波变换后的数据
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # 只取近似系数
    coeffs = [coeffs[0]]
    return pywt.waverec(coeffs, wavelet)

# 示例：对单个样本进行小波变换
# sample_data = np.random.randn(4, 512)  # 假设样本
# transformed_data = wavelet_transform(sample_data)
