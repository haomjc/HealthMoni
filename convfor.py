import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, src2, src2)[0])
        src2 = self.norm2(src)
        src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src2)))))
        return src


class Convformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, embed_dim, num_heads, num_layers, time_series_length):
        super(Convformer, self).__init__()
        self.conv1 = ConvBlock(in_channels, embed_dim)
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim * time_series_length, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(2, 0, 1)  # Change to (seq_len, batch, embed_dim)
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = x.permute(1, 2, 0)  # Back to (batch, embed_dim, seq_len)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc(x)
        return x

# 示例使用
# net = Convformer(in_channels=4, out_channels=10, num_classes=28, embed_dim=40, num_heads=4, num_layers=2, time_series_length=4096)
# input_tensor = torch.randn((batch_size, 4, 4096))
# output = net(input_tensor)
