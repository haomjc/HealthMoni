import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        if self.downsample is not None:
            residual = self.downsample(x)

        # 确保残差和输出尺寸相匹配
        if out.size(2) != residual.size(2):
            out = F.pad(out, (0, residual.size(2) - out.size(2)))

        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, num_channels, kernel_size, dropout, num_classes):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 4 if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1]  # 取最后一个时间步
        x = self.fc(x)
        return x

# 示例使用
# net = EnhancedTCN(num_channels=[32, 64, 128], kernel_size=3, dropout=0.3, num_classes=28)
# input_tensor = torch.randn((batch_size, 4, time_series_length))  # 假设输入形状
# output = net(input_tensor)
