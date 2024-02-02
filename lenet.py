import torch
import torch.nn as nn
import torch.nn.functional as F

time_series_length = 4096


class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.relu(x)
        return x


class LeNet1D(nn.Module):
    def __init__(self, num_classes=28):
        super(LeNet1D, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, 7)  # 增加通道数
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2, 2)
        self.resblock1 = EnhancedResidualBlock(16, 32, 5, 1, 2)  # 增加通道数
        self.resblock2 = EnhancedResidualBlock(32, 64, 5, 1, 2)  # 增加通道数
        self.resblock3 = EnhancedResidualBlock(64, 128, 5, 1, 2)  # 新增额外的残差块

        # 计算卷积层输出尺寸
        self._to_linear = None
        self.calculate_to_linear(4, time_series_length)

        self.fc1 = nn.Linear(self._to_linear, 256)  # 增加神经元数量
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)  # 增加神经元数量
        self.fc3 = nn.Linear(128, num_classes)

    def calculate_to_linear(self, channels, seq_length):
        with torch.no_grad():
            input = torch.randn(1, channels, seq_length)
            output = self.pool(F.relu(self.bn1(self.conv1(input))))
            output = self.resblock1(output)
            output = self.resblock2(output)
            output = self.resblock3(output)  # 通过额外的残差块
            self._to_linear = output.numel() // output.shape[0]

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)  # 通过额外的残差块
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
