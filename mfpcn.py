import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FourierLayer(nn.Module):
    def __init__(self, time_series_length):
        super(FourierLayer, self).__init__()
        self.time_series_length = time_series_length

    def forward(self, x):
        # 假设 x 的形状为 (batch_size, channels, time_series_length)
        # 进行傅里叶变换并分离实部和虚部
        fft_result = torch.fft.rfft(x)
        real_part = fft_result.real
        imag_part = fft_result.imag

        # 将实部和虚部合并在一起，形成新的特征维度
        combined = torch.cat((real_part, imag_part), dim=-1)

        return combined


class MFPCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, time_series_length):
        super(MFPCN, self).__init__()
        self.fourier = FourierLayer(time_series_length)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(out_channels, out_channels * 2, kernel_size=3, padding=1)  # 新增卷积层
        self.bn2 = nn.BatchNorm1d(out_channels * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # 自适应池化
        self.fc = nn.Linear(out_channels * 2, num_classes)  # 调整全连接层的输入

    def forward(self, x):
        x = self.fourier(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)  # 通过新增的卷积层
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.adaptive_pool(x)  # 应用自适应池化
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
