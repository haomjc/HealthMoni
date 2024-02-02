import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, \
    f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from datasave import train_loader, test_loader  # 请确保这些加载器是正确的

from sklearn.model_selection import train_test_split


# 序列长度最多2048

def extract_features(data_loader):
    # 这里添加您的特征提取逻辑
    # 示例代码
    features, labels = [], []
    for data, label in data_loader:
        # 假设data是您的信号数据
        flattened_data = data.reshape(data.shape[0], -1)
        features.append(flattened_data)
        labels.append(label)

    return np.vstack(features), np.concatenate(labels)


# G-mean 函数
def g_mean(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return np.prod(sensitivity) ** (1 / len(sensitivity))


def train_and_evaluate(X_train, X_test, y_train, y_test, parameters, scaler):
    # 归一化
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练SVM模型
    clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    model = clf.best_estimator_
    y_pred = model.predict(X_test_scaled)

    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    f_measure = f1_score(y_test, y_pred, average='weighted')
    gmean = g_mean(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "f_measure": f_measure,
        "g_mean": gmean
    }


# 运行多次训练和评估
num_runs = 10
results = []
parameters = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'poly']}
scaler = StandardScaler()

X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)


def combine_loaders(loader1, loader2):
    X1, y1 = extract_features(loader1)
    X2, y2 = extract_features(loader2)
    return np.vstack([X1, X2]), np.concatenate([y1, y2])


# 合并来自两个 DataLoader 的数据
X, y = combine_loaders(train_loader, test_loader)

for i in range(num_runs):
    print(f"Running iteration {i + 1}/{num_runs}...")

    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    run_result = train_and_evaluate(X_train, X_test, y_train, y_test, parameters, scaler)
    results.append(run_result)

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 保存到Excel文件
results_df.to_excel("svm_performance_metrics.xlsx", index=False)
