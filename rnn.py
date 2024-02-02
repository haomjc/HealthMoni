time_series_length = 4096

import torch
import torch.nn as nn
import torch.nn.functional as F


class rnn_Net(nn.Module):
    def __init__(self, input_size=time_series_length, hidden_size=256, num_layers=5, num_classes=28, dropout_rate=0.5):
        super(rnn_Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 一维卷积层，用于初步特征提取
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)  # 修改 in_channels=4
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)

        # 双向GRU层
        self.gru = nn.GRU(time_series_length // 2, hidden_size, num_layers,
                          batch_first=True, dropout=dropout_rate, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向GRU的输出维度是hidden_size的两倍

        # 正则化
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 卷积层
        x = x.squeeze(1)  # 假设数据的形状是 [batch_size, 1, channels, length]

        x = self.pool(F.relu(self.bn1(self.conv2(F.relu(self.conv1(x))))))

        # RNN层
        x = x.squeeze(2)  # 去除多余的维度
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.gru(x, h0)

        # 取最后一个时间步的输出
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])

        return x
