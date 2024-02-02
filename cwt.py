import torch
import torch.nn as nn
import numpy as np
from datasave import train_loader, test_loader
from early_stopping import EarlyStopping
from label_smoothing import LSR
from oneD_Meta_ACON import MetaAconC
import time
from torchsummary import summary
from adabn import reset_bn, fix_bn

import visdom

# 四组数据，目前只用了一组（四组都已使用）
time_series_length = 512
GRU_length = 60


# 512 1024 2048 4096 8192 16384
# 60 124 252 508 1020 2044

# 一维数据通过小波变换处理？？

# 又怎么绘制tsne和混淆矩阵

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


#
setup_seed(1)

# class swish(nn.Module):
#     def __init__(self):
#         super(swish, self).__init__()
#
#     def forward(self, x):
#         x = x * F.sigmoid(x)
#         return x
# def reset_bn(module):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         module.track_running_stats = False
# def fix_bn(module):
#     if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
#         module.track_running_stats = True

# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
# import pywt
# import torch


# import numpy as np
# import torch.nn as nn

# # 示例：定义一个CWT变换函数
# def apply_cwt(data, scales, wavelet_name='morl'):
#     coefficients = []
#     for sample in data:
#         coef, _ = pywt.cwt(sample, scales, wavelet_name)
#         coefficients.append(coef)
#     return np.array(coefficients)
#
#
# # 示例：在数据加载或预处理阶段应用CWT
# # 假设 data 是原始数据
# scales = range(1, 128)  # 选择合适的尺度
# transformed_data = apply_cwt(data, scales)
#
# # 将变换后的数据转换为适合PyTorch模型的格式
# transformed_data_tensor = torch.tensor(transformed_data, dtype=torch.float32)
#
#
# # 然后，您可以将这个张量作为输入传递给您的模型
#
# # 注意：您可能需要根据小波变换后的数据形状调整模型结构


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # self.pool_w = nn.AdaptiveAvgPool1d(1)
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        mip = max(6, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(mip, track_running_stats=False)
        self.act = MetaAconC(mip)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(4, 50, kernel_size=18, stride=2),
                                  nn.BatchNorm1d(50),
                                  MetaAconC(50))
        self.p1_2 = nn.Sequential(nn.Conv1d(50, 30, kernel_size=10, stride=2),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p1_3 = nn.MaxPool1d(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(4, 50, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(50),
                                  MetaAconC(50))
        self.p2_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(40),
                                  MetaAconC(40))
        self.p2_3 = nn.MaxPool1d(2, 2)
        self.p2_4 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=6, stride=1),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p2_5 = nn.Sequential(nn.Conv1d(30, 30, kernel_size=6, stride=2),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p2_6 = nn.MaxPool1d(2, 2)
        self.p3_0 = CoordAtt(30, 30)
        # self.p3_1 = nn.Sequential(nn.GRU(124, 64, bidirectional=True))  #
        self.p3_1 = nn.Sequential(nn.GRU(GRU_length, 64, bidirectional=True))  #

        # self.p3_2 = nn.Sequential(nn.LSTM(128, 512))
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.p4 = nn.Sequential(nn.Linear(30, 28))

    def forward(self, x):
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        encode = torch.mul(p1, p2)
        # p3 = self.p3_2(self.p3_1(encode))
        p3_0 = self.p3_0(encode).permute(1, 0, 2)
        p3_2, _ = self.p3_1(p3_0)
        # p3_2, _ = self.p3_2(p3_1)
        p3_11 = p3_2.permute(1, 0, 2)  #
        p3_12 = self.p3_3(p3_11).squeeze()
        # p3_11 = h1.permute(1,0,2)
        # p3 = self.p3(encode)
        # p3 = p3.squeeze()
        # p4 = self.p4(p3_11)  # LSTM(seq_len, batch, input_size)
        # p4 = self.p4(encode)
        p4 = self.p4(p3_12)
        return p4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
# model.load_state_dict(torch.load('./data7/B0503_AdamP_AMS_Nb.pt'))
# for m in model.modules():
#     if isinstance(m, nn.Conv1d):
#         #nn.init.normal_(m.weight)
#         #nn.init.xavier_normal_(m.weight)
#         nn.init.kaiming_normal_(m.weight)
#         #nn.init.constant_(m.bias, 0)
#     # elif isinstance(m, nn.GRU):
#     #     for param in m.parameters():
#     #         if len(param.shape) >= 2:
#     #             nn.init.orthogonal_(param.data)
#     #         else:
#     #             nn.init.normal_(param.data)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0, std=torch.sqrt(torch.tensor(1/30)))
# input = torch.rand(20, 1, 1024).to(device)
# output = model(input)
# print(output.size())
# with SummaryWriter(log_dir='logs', comment='Net') as w:
#      w.add_graph(model, (input,))
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', 'logs'])
# url = tb.launch()
summary(model, input_size=(4, time_series_length))
# criterion = nn.CrossEntropyLoss()

criterion = LSR()
# criterion = CrossEntropyLoss_LSR(device)
# from adabound import AdaBound
# optimizer = AdaBound(model.parameters(), lr=0.001, weight_decay=0.0001, amsbound=True)
# from EAdam import EAdam
# optimizer = EAdam(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.000, weight_decay=0.0001)
bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},
              {'parameters': others_list}]
# optimizer = Nadam(model.parameters())
# optimizer = RAdam(model.parameters())
# from torch_optimizer import AdamP
# from adamp import AdamP
from AdamP_amsgrad import AdamP

optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=0.0001, nesterov=True, amsgrad=True)
# from adabelief_pytorch import AdaBelief
# optimizer = AdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
# from ranger_adabelief import RangerAdaBelief
# optimizer = RangerAdaBelief(model.parameters(), lr=0.001, weight_decay=0.0001, weight_decouple=True)
losses = []
acces = []
eval_losses = []
eval_acces = []
early_stopping = EarlyStopping(patience=10, verbose=True)
starttime = time.time()
for epoch in range(150):
    train_loss = 0
    train_acc = 0
    model.train()
    for img, label in train_loader:
        img = img.float()
        img = img.to(device)
        # label = (np.argmax(label, axis=1)+1).reshape(-1, 1)
        # label=label.float()

        label = label.to(device)
        label = label.long()
        out = model(img)
        out = torch.squeeze(out).float()
        # label=torch.squeeze(label)

        # out_1d = out.reshape(-1)
        # label_1d = label.reshape(-1)
        # print(out, label)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(scheduler.get_lr())
        train_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))

    # 这里是验证集的位置
    eval_loss = 0
    eval_acc = 0
    model.eval()
    model.apply(reset_bn)
    for img, label in test_loader:
        img = img.type(torch.FloatTensor)
        img = img.to(device)
        label = label.to(device)
        label = label.long()

        # 新加的，是否可用
        # label = label.unsqueeze(1)

        # img = img.view(img.size(0), -1)
        out = model(img)
        out = torch.squeeze(out).float()

        # out = out.unsqueeze(1)

        # print(out, '\n\n', label)

        loss = criterion(out, label).sum(dim=-1).mean()
        #
        eval_loss += loss.item()
        #
        _, pred = out.max(1)

        # print(pred, "pred")

        num_correct = (pred == label).sum().item()
        # print((pred == label).sum())

        # print(pred, '\n\n', label)
        acc = num_correct / img.shape[0]

        # print(num_correct, img.shape)

        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
                  eval_loss / len(test_loader), eval_acc / len(test_loader)))

    # 使用visdom对损失和准确率进行可视化
    vis = visdom.Visdom()
    vis.line(Y=np.array(losses), X=np.array(range(len(losses))), win='train_loss', opts={'title': 'train_loss'})
    vis.line(Y=np.array(acces), X=np.array(range(len(acces))), win='train_acc', opts={'title': 'train_acc'})
    vis.line(Y=np.array(eval_losses), X=np.array(range(len(eval_losses))), win='test_loss',
             opts={'title': 'test_loss'})
    vis.line(Y=np.array(eval_acces), X=np.array(range(len(eval_acces))), win='test_acc', opts={'title': 'test_acc'})

    early_stopping(eval_loss / len(test_loader), model)
    model.apply(fix_bn)
    if early_stopping.early_stop:
        print("Early stopping")
        break
endtime = time.time()
dtime = endtime - starttime
print("time：%.8s s" % dtime)
torch.save(model.state_dict(), '\B0503_LSTM.pt')
import pandas as pd

pd.set_option('display.max_columns', None)  # 
pd.set_option('display.max_rows', None)  #

# add confusion matrix calculation
from sklearn.metrics import confusion_matrix
import numpy as np

# Get predictions on test set
model.eval()
Y_test = []
y_pred = []

with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        y_test_pred = model(X_test.float())
        # print(y_test_pred)

        y_pred.extend(y_test_pred.argmax(1).cpu().numpy())
        y_test = y_test.cpu().numpy()
        Y_test.extend(y_test)
# Calculate confusion matrix
# print(Y_test, y_pred)
conf_mat = confusion_matrix(Y_test, y_pred)
# print(conf_mat)

conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
conf_mat_norm = np.around(conf_mat_norm, decimals=2)

# 将其转换为DataFrame
conf_df = pd.DataFrame(conf_mat_norm, columns=range(28), index=range(28))

# 保存到excel
conf_df.to_excel("confusion_matrix.xlsx")

# visualize the confusion matrix
import matplotlib.pyplot as plt

plt.imshow(conf_mat_norm, interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()
plt.xticks(np.arange(28))
plt.yticks(np.arange(28))
plt.show()

# t-SNE可视化的代码,用于将高维数据投影到2D空间进行可视化
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 将输出结果和标签取出
outputs = []
labels = []
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        outputs.append(model(data.float()))
        labels.append(label)

outputs = torch.cat(outputs, dim=0).cpu().numpy()
labels = torch.cat(labels, dim=0).cpu().numpy()

# # 进行t-SNE降维
# tsne = TSNE(n_components=2, learning_rate=100).fit_transform(outputs)
#
# tsne_df = pd.DataFrame(tsne, columns=["x", "y"])
# tsne_df["label"] = labels
#
# tsne_df.to_excel("tsne.xlsx")
#
# # 可视化
# plt.figure(figsize=(5, 5))
# plt.xticks([])
# plt.yticks([])
#
# plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis')
# plt.colorbar()  # 添加颜色条，显示映射关系
#
# plt.savefig('tsne.png')
# plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具包
import pandas as pd

# 你之前的代码部分保持不变

# 进行t-SNE降维
tsne = TSNE(n_components=3, learning_rate=100).fit_transform(outputs)  # 改为3维

tsne_df = pd.DataFrame(tsne, columns=["x", "y", "z"])  # 添加第三个维度
tsne_df["label"] = labels

tsne_df.to_excel("tsne_3d.xlsx")

# 可视化
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')  # 设置为3D图

# 绘制散点图
sc = ax.scatter(tsne_df['x'], tsne_df['y'], tsne_df['z'], c=labels, cmap='viridis')

plt.colorbar(sc)  # 添加颜色条
plt.savefig('tsne_3d.png')
plt.show()
