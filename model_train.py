import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, confusion_matrix
from early_stopping import EarlyStopping
from label_smoothing import LSR, AdaptiveLabelSmoothing
from oneD_Meta_ACON import MetaAconC
import time
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
# from adabn import reset_bn, fix_bn
from adabn import AdaptiveBN
import visdom
from fvcore.nn import FlopCountAnalysis

from dwtc_bigru import dwtc_bigru_Net
from dcgan import dcgan_Net
from wdcnn import wdcnn_Net
from rnn import rnn_Net
from tfn import TimeSeriesTransformer
from lstm import LSTMModel
from resnet import ResNet1D
# from dwtc_bigru import CoordAtt
from lenet import LeNet1D
from incep import InceptionNet
from tcn import TCN
from convfor import Convformer
from mfpcn import MFPCN

model_name = "incep"  # 从参数或配置文件获取

if model_name == "dwtc_bigru":
    from datasave_dwt import train_loader, test_loader
else:
    from datasave import train_loader, test_loader

if model_name == "dwtc_bigru":
    Net = dwtc_bigru_Net

# elif model_name == "dcgan":
#     Net = dcgan_Net

elif model_name == "wdcnn":
    Net = wdcnn_Net  # 使用WDCNN模型

elif model_name == "rnn":
    Net = rnn_Net

# elif model_name == "tfn":
#     Net = TimeSeriesTransformer

# 目前lstm效果不好
# elif model_name == "lstm":
#     Net = LSTMModel

# elif model_name == "resnet":
#     Net = ResNet1D

elif model_name == "lenet":
    Net = LeNet1D

elif model_name == "incep":
    Net = InceptionNet

# elif model_name == "tcn":
#     Net = TCN

# elif model_name == "convfor":
#     Net = Convformer

elif model_name == "mfpcn":
    Net = MFPCN

# 四组数据，目前只用了一组（四组都已使用）
time_series_length = 4096

# 512 1024 2048 4096 8192 16384
# 60 124 252 508 1020 2044

# 假设的参数值，您需要根据您的模型和数据来调整它们
nz = 100  # 隐空间向量的大小，通常是一个根据经验选取的值
ngf = 64  # 生成器特征映射的大小
ndf = 64  # 判别器特征映射的大小
nc = 4  # 输入数据的通道数，根据您的数据设置


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(3407)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model_name == "dwtc_bigru":
    model = Net().to(device)  # 其他必要的参数

# elif model_name == "dcgan":
#     model = Net(nz=nz, ngf=ngf, ndf=ndf, nc=nc).to(device)

elif model_name == "wdcnn":
    model = Net().to(device)

elif model_name == "rnn":
    model = Net().to(device)

# elif model_name == "tfn":
#     model = Net(input_dim=4, num_classes=28, seq_length=time_series_length, d_model=6, nhead=2, num_layers=3,
#                 dropout=0.1).to(device)

# elif model_name == "lstm":
#     model = Net(input_size=time_series_length, hidden_size=1024, num_layers=3, num_classes=28).to(device)


# resnet实现效果不好
# elif model_name == "resnet":
#     model = Net(num_classes=28).to(device)

elif model_name == "lenet":
    model = Net(num_classes=28).to(device)

elif model_name == "incep":
    model = Net(num_classes=28).to(device)

# elif model_name == "tcn":
#     model = Net(num_channels=[32, 64, 128], kernel_size=3, dropout=0.3, num_classes=28).to(device)

# elif model_name == "convfor":
#     model = Net(in_channels=4, out_channels=10, num_classes=28, embed_dim=4, num_heads=2, num_layers=1,
#                 time_series_length=time_series_length).to(device)

elif model_name == "mfpcn":
    model = Net(in_channels=4, out_channels=16, num_classes=28, time_series_length=time_series_length).to(device)

if model_name == "tfn":
    pass
elif model_name == "lstm":
    pass
else:
    summary(model, input_size=(4, time_series_length))

if model_name == "lstm" or model_name == "tfn":
    pass
else:
    # criterion = nn.CrossEntropyLoss()

    # 假设您的模型被命名为 `model`，并且有一个名为 `dummy_input` 的输入
    # dummy_input = torch.randn(batch_size, channels, height, width)
    dummy_input = torch.randn(128, 4, time_series_length).to(device)
    # 使用 `FlopCountAnalysis` 分析模型
    flops = FlopCountAnalysis(model, dummy_input)
    print('FLOPs: ', flops.total())

    writer = SummaryWriter('runs/model_visualization')

    # 假设您的模型是 'model'，并且您有一个名为 'dummy_input' 的输入样本
    writer.add_graph(model, dummy_input)
    writer.close()

# criterion = LSR()
criterion = AdaptiveLabelSmoothing(num_classes=28, initial_e=0.1, reduction='mean')

bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
parameters = [{'parameters': bias_list, 'weight_decay': 0},
              {'parameters': others_list}]

from AdamP_amsgrad import AdamP

# optimizer = AdamP(model.parameters(), lr=0.001, weight_decay=0.0001, nesterov=True, amsgrad=True)
optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=0.0001)

losses = []
acces = []
eval_losses = []
eval_acces = []

early_stopping = EarlyStopping(patience=10, verbose=True)
starttime = time.time()

# 初始化AdaptiveBN
adabn = AdaptiveBN(model)

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
    # model.apply(reset_bn)
    # 保存原始BN层统计数据
    adabn.save_original_stats()

    # 使用AdaptiveBN重置并更新BN统计
    adabn.update_stats_with_target_data(test_loader)

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

        loss = criterion(out, label).sum(dim=-1).mean()
        #
        eval_loss += loss.item()
        #
        _, pred = out.max(1)

        # print(pred, "pred")

        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('轮次: {}, 训练损失: {:.4f}, 训练准确率: {:.4f}, 测试损失: {:.4f}, 测试准确率: {:.4f}'
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

    # 恢复原始BN层统计数据
    adabn.restore_original_stats()
    # model.apply(fix_bn)
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

import pandas as pd

loss_df = pd.DataFrame({
    'Training Loss': losses,
    'Evaluation Loss': eval_losses,
    'Training Accu': acces,
    'Evaluation Accu': eval_acces
})

loss_df.to_excel("training_evaluation_loss_eval.xlsx", index_label='Epoch')

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


# 计算 R square, MAE, RMSE, MSE, G mean 和 F measure
def g_mean(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return np.prod(sensitivity) ** (1 / len(sensitivity))


def f_measure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


# 假设 Y_test 和 y_pred 已经包含了真实标签和预测结果
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)
g_mean_value = g_mean(Y_test, y_pred)
f_measure_value = f_measure(Y_test, y_pred)

# 输出结果
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
print(f"G mean: {g_mean_value}")
print(f"F measure: {f_measure_value}")

# t-SNE可视化的代码,用于将高维数据投影到2D空间进行可视化
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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具包
import pandas as pd

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
