import torch
import torch.nn as nn

import torch
import torch.nn as nn


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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义残差块
        self.resblock1 = EnhancedResidualBlock(input_size, 32, 5, 1, 2)
        self.resblock2 = EnhancedResidualBlock(32, 64, 5, 1, 2)
        self.resblock3 = EnhancedResidualBlock(64, 128, 5, 1, 2)

        # 定义 LSTM 层
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True)

        # 定义输出层
        self.fc1 = nn.Linear(hidden_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 通过残差块
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        # 重塑数据以适应 LSTM 层
        x = x.transpose(1, 2)  # 转换为 (batch, seq_len, features)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        x, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        x = x[:, -1, :]

        # 通过全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
