import torch
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, confusion_matrix
from fvcore.nn import FlopCountAnalysis
import pandas as pd

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, confusion_matrix
from early_stopping import EarlyStopping
from label_smoothing import LSR, AdaptiveLabelSmoothing
from oneD_Meta_ACON import MetaAconC
import time
import numpy as np

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


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, time_series_length, epochs=150):
    starttime = time.time()

    # 计算FLOPs
    dummy_input = torch.randn(128, 4, time_series_length).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()

    losses, acces, eval_losses, eval_acces = [], [], [], []
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(epochs):
        train_loss, train_acc = 0, 0
        model.train()
        for img, label in train_loader:
            img, label = img.float().to(device), label.long().to(device)
            out = model(img).squeeze().float()
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = out.max(1)
            train_acc += (pred == label).sum().item() / img.shape[0]

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

        # 验证集评估
        eval_loss, eval_acc = 0, 0
        model.eval()

        # 保存原始BN层统计数据
        adabn.save_original_stats()

        # 使用AdaptiveBN重置并更新BN统计
        adabn.update_stats_with_target_data(test_loader)

        for img, label in test_loader:
            img, label = img.float().to(device), label.long().to(device)
            out = model(img).squeeze().float()
            loss = criterion(out, label).sum(dim=-1).mean()
            eval_loss += loss.item()
            _, pred = out.max(1)
            eval_acc += (pred == label).sum().item() / img.shape[0]

        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))

        early_stopping(eval_loss / len(test_loader), model)

        # 恢复原始BN层统计数据
        adabn.restore_original_stats()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 性能指标计算
    Y_test, y_pred = [], []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.float().to(device), y_test.to(device)
            y_test_pred = model(X_test)
            y_pred.extend(y_test_pred.argmax(1).cpu().numpy())
            Y_test.extend(y_test.cpu().numpy())

    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    g_mean_value = np.prod(
        (confusion_matrix(Y_test, y_pred).diagonal() / confusion_matrix(Y_test, y_pred).sum(axis=1))) ** (
                           1 / len(np.unique(Y_test)))
    f_measure_value = f1_score(Y_test, y_pred, average='weighted')

    endtime = time.time()
    dtime = endtime - starttime

    # 将结果保存到字典中
    results = {
        'train_loss': losses[-1],
        'train_accuracy': acces[-1],
        'eval_loss': eval_losses[-1],
        'eval_accuracy': eval_acces[-1],
        'time': dtime,
        'FLOPs': total_flops,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2_Score': r2,
        'G_mean': g_mean_value,
        'F_measure': f_measure_value
    }

    return results


# 模型创建函数示例
def create_model(model_name):
    if model_name == "dwtc_bigru":
        return dwtc_bigru_Net()  # 返回模型实例
    elif model_name == "wdcnn":
        return wdcnn_Net()

    elif model_name == "rnn":
        return rnn_Net()

    elif model_name == "lenet":
        return LeNet1D(num_classes=28)

    elif model_name == "incep":
        return InceptionNet(num_classes=28)

    elif model_name == "mfpcn":
        return MFPCN(in_channels=4, out_channels=16, num_classes=28, time_series_length=time_series_length)

    else:
        raise ValueError("Unknown model name")


# 参数
model_names = ["dwtc_bigru", "wdcnn", "rnn", "lenet", "incep", "mfpcn"]
num_runs = 10
results = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_series_length = 4096

# 初始化一个空的DataFrame用于存储结果
results_df = pd.DataFrame()

for model_name in model_names:

    if model_name == "dwtc_bigru":
        from datasave_dwt import train_loader, test_loader
    else:
        from datasave import train_loader, test_loader

    for run in range(num_runs):
        model = create_model(model_name).to(device)
        # 初始化AdaptiveBN
        adabn = AdaptiveBN(model)

        criterion = AdaptiveLabelSmoothing(num_classes=28, initial_e=0.1, reduction='mean')

        bias_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias')
        others_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias')
        parameters = [{'parameters': bias_list, 'weight_decay': 0},
                      {'parameters': others_list}]

        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=0.0001)

        run_result = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device,
                                        time_series_length=time_series_length, epochs=150
                                        )
        run_result['model'] = model_name
        run_result['run'] = run + 1

        # 添加结果到DataFrame并立即保存
        results_df = results_df._append(run_result, ignore_index=True)
        results_df.to_excel("model_performance_metrics.xlsx", index=False)
