import torch
import torch.nn as nn
from oneD_Meta_ACON import MetaAconC
from Mish import MemoryEfficientMish

time_series_length = 4096
GRU_length = 508


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # self.pool_w = nn.AdaptiveAvgPool1d(1)
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        mip = max(6, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip, track_running_stats=False)
        # self.act = MetaAconC(mip)
        self.act = MemoryEfficientMish()

        self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, w = x.size()
        x_w = self.pool_w(x)
        y = torch.cat([identity, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_ww, x_c = torch.split(y, [w, 1], dim=2)
        a_w = self.conv_w(x_ww)
        a_w = a_w.sigmoid()
        out = identity * a_w
        return out


class dwtc_bigru_Net(nn.Module):
    def __init__(self):
        super(dwtc_bigru_Net, self).__init__()
        self.p0_1 = nn.Sequential(nn.Conv1d(4, 50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(50),
                                  # MetaAconC(50))
                                  MemoryEfficientMish())
        self.p0_2 = nn.Sequential(nn.Conv1d(50, 30, kernel_size=10, stride=2),
                                  nn.BatchNorm1d(30),
                                  # MetaAconC(30))
                                  MemoryEfficientMish())
        self.p0_3 = nn.MaxPool1d(2, 2)
        # self.p0_3 = nn.AdaptiveMaxPool1d(GRU_length)

        self.p1_1 = nn.Sequential(nn.Conv1d(4, 50, kernel_size=14, stride=1),
                                  nn.BatchNorm1d(50),
                                  # MetaAconC(50))
                                  MemoryEfficientMish())
        self.p1_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=12, stride=1),
                                  nn.BatchNorm1d(40),
                                  # MetaAconC(40))
                                  MemoryEfficientMish())
        # self.p1_3 = nn.MaxPool1d(2, 2)
        self.p1_3 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=10, stride=1),
                                  nn.BatchNorm1d(30),
                                  # MetaAconC(30))
                                  MemoryEfficientMish())

        # self.p1_5 = nn.MaxPool1d(2, 2)
        self.p1_4 = nn.AdaptiveMaxPool1d(GRU_length)

        self.p2_1 = nn.Sequential(nn.Conv1d(4, 50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(50),
                                  # MetaAconC(50))
                                  MemoryEfficientMish())
        self.p2_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(40),
                                  # MetaAconC(40))
                                  MemoryEfficientMish())
        self.p2_3 = nn.MaxPool1d(2, 2)
        self.p2_4 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(30),
                                  # MetaAconC(30))
                                  MemoryEfficientMish())
        self.p2_5 = nn.Sequential(nn.Conv1d(30, 30, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(30),
                                  # MetaAconC(30))
                                  MemoryEfficientMish())
        self.p2_6 = nn.MaxPool1d(2, 2)
        # self.p2_6 = nn.AdaptiveMaxPool1d(GRU_length)

        self.p3_0 = CoordAtt(30, 30)
        # self.p3_1 = nn.Sequential(nn.GRU(124, 64, bidirectional=True))  #
        self.p3_1 = nn.Sequential(nn.GRU(GRU_length, 64, bidirectional=True))  #

        # self.p3_2 = nn.Sequential(nn.LSTM(128, 512))
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(30, 28))

    def forward(self, x):
        p0 = self.p0_3(self.p0_2(self.p0_1(x)))
        p1 = self.p1_4(self.p1_3(self.p1_2(self.p1_1(x))))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))

        # print(f"p0 size: {p0.size()}")
        # print(f"p1 size: {p1.size()}")
        # print(f"p2 size: {p2.size()}")

        encode = torch.mul(torch.mul(p0, p1), p2)
        # p3 = self.p3_2(self.p3_1(encode))
        p3_0 = self.p3_0(encode).permute(1, 0, 2)
        p3_2, _ = self.p3_1(p3_0)
        # p3_2, _ = self.p3_2(p3_1)
        p3_11 = p3_2.permute(1, 0, 2)  #
        p3_12 = self.p3_3(p3_11).squeeze()

        p4 = self.p4(p3_12)
        return p4
