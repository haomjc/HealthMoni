import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet1D(nn.Module):
    def __init__(self, num_classes=28):
        super(ResNet1D, self).__init__()
        # 加载预定义的 ResNet50 模型
        self.model = resnet50(pretrained=True)

        # 修改第一个卷积层以适应一维数据，假设输入数据有4个通道
        self.model.conv1 = nn.Conv1d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改批量归一化层为一维
        self.model.bn1 = nn.BatchNorm1d(64)

        # 替换全连接层以适应新的分类数量
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
