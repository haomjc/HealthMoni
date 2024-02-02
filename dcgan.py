import torch
import torch.nn as nn
from oneD_Meta_ACON import MetaAconC


# time_series_length = 512
# GRU_length = 60


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入: nz x 1 x 1
            nn.ConvTranspose1d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # 状态大小: (ngf*4) x 4
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # 状态大小: (ngf*2) x 8
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # 状态大小: (ngf) x 16
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 最终状态大小: (nc) x 32
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: nc x 512
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf) x 256
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf*2) x 128
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf*4) x 64
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小: (ndf*8) x 32
            nn.Conv1d(ndf * 8, 1, 32, 1, 0, bias=False),
            nn.Sigmoid()
            # 最终状态大小: 1 x 1 x 1
        )

    def forward(self, input):
        return self.main(input)


class dcgan_Net(nn.Module):

    def __init__(self, nz, ngf, ndf, nc):
        super(dcgan_Net, self).__init__()
        self.generator = Generator(nz, ngf, nc)
        self.discriminator = Discriminator(nc, ndf)

    # def forward(self, input):
    #     return self.main(input)

# 以上的程序是正确的吗？
