import torch
import torch.nn as nn
import torch.nn.functional as F

time_series_length = 4096


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        # 分支的输出通道数
        branch_out_channels = out_channels // 4

        self.branch1x1 = nn.Conv1d(in_channels, branch_out_channels, kernel_size=1)

        self.branch5x5_1 = nn.Conv1d(in_channels, branch_out_channels, kernel_size=1)
        self.branch5x5_2 = nn.Conv1d(branch_out_channels, branch_out_channels, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv1d(in_channels, branch_out_channels, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv1d(branch_out_channels, branch_out_channels, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv1d(branch_out_channels, branch_out_channels, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv1d(in_channels, branch_out_channels, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionNet(nn.Module):
    def __init__(self, num_classes=28):
        super(InceptionNet, self).__init__()
        self.conv1 = nn.Conv1d(4, 16, kernel_size=3, padding=1)  # 调整卷积核大小
        self.bn1 = nn.BatchNorm1d(16)  # 添加批量归一化
        self.relu1 = nn.ReLU()
        self.inception1 = InceptionModule(16, 64)
        self.inception2 = InceptionModule(64, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 使用全局平均池化
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 示例使用
# net = InceptionNet(num_classes=28)
# input_tensor = torch.randn((batch_size, 4, time_series_length))  # 假设输入形状
# output = net(input_tensor)
