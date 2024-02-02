import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, f1_score
from datasave import train_loader, test_loader


def extract_features(data_loader):
    features, labels = [], []
    for data, label in data_loader:
        # 假设data是您的信号数据，需要将其转换为一维特征向量
        # 这里可以添加您的特征提取逻辑
        flattened_data = data.reshape(data.shape[0], -1)
        features.append(flattened_data)
        labels.append(label)

    return np.vstack(features), np.concatenate(labels)


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    f_measure = f1_score(y_test, y_pred, average='weighted')
    gmean = g_mean(y_test, y_pred)
    return accuracy, mae, mse, rmse, f_measure, gmean


def g_mean(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return np.prod(sensitivity) ** (1 / len(sensitivity))


# 运行多次训练和评估
num_runs = 10
results = []

for run in range(num_runs):
    print(f"Running iteration {run + 1}/{num_runs}...")

    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    run_results = evaluate_model(y_test, y_pred)
    results.append(run_results)

# 将结果转换为DataFrame
results_df = pd.DataFrame(results, columns=['Accuracy', 'MAE', 'MSE', 'RMSE', 'F-Measure', 'G-Mean'])

# 保存到Excel文件
results_df.to_excel("random_forest_performance_metrics.xlsx", index=False)
