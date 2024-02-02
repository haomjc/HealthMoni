import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, seq_length, d_model, nhead, num_layers, dropout):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length

        # 添加改进的卷积层
        self.conv1 = nn.Conv1d(4, 16, 7)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2, 2)
        self.resblock1 = EnhancedResidualBlock(16, 32, 5, 1, 2)
        self.resblock2 = EnhancedResidualBlock(32, 64, 5, 1, 2)
        self.resblock3 = EnhancedResidualBlock(64, 128, 5, 1, 2)

        # 计算卷积层输出尺寸
        self._to_linear = None
        self.calculate_to_linear(4, seq_length)

        # 嵌入层
        self.embedding = nn.Linear(self._to_linear, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.out = nn.Linear(d_model * self._to_linear, num_classes)

    def calculate_to_linear(self, input_dim, seq_length):
        # 使用一个虚拟输入来计算经过卷积层之后的数据维度
        dummy_input = torch.randn(1, input_dim, seq_length)
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.resblock1(dummy_output)
        dummy_output = self.resblock2(dummy_output)
        dummy_output = self.resblock3(dummy_output)
        self._to_linear = dummy_output.numel() // dummy_output.shape[0]

        with torch.no_grad():
            input = torch.randn(1, input_dim, seq_length)
            output = self.pool(F.relu(self.bn1(self.conv1(input))))
            output = self.resblock1(output)
            output = self.resblock2(output)
            output = self.resblock3(output)
            self._to_linear = output.numel() // output.shape[0]

    def forward(self, src):
        # 应用卷积层和残差块
        src = self.pool(F.relu(self.bn1(self.conv1(src))))
        src = self.resblock1(src)
        src = self.resblock2(src)
        src = self.resblock3(src)
        src = src.view(-1, self._to_linear)

        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # 调整形状以适应 Transformer 编码器
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2).flatten(start_dim=1)

        output = F.dropout(output, p=0.5)
        output = self.out(output)
        return F.log_softmax(output, dim=-1)
