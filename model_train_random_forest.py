import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from datasave import train_loader, test_loader


# 特征提取函数（根据需要进行调整）
def extract_features(data_loader):
    features, labels = [], []
    for data, label in data_loader:
        # 假设data是您的信号数据，需要将其转换为一维特征向量
        # 这里可以添加您的特征提取逻辑
        flattened_data = data.reshape(data.shape[0], -1)
        features.append(flattened_data)
        labels.append(label)

    return np.vstack(features), np.concatenate(labels)


# 提取训练和测试数据的特征
X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# 使用随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 测试模型性能
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

# 输出准确率和混淆矩阵
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_mat)

# 可视化混淆矩阵
conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
plt.imshow(conf_mat_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xticks(np.arange(len(np.unique(y_test))))
plt.yticks(np.arange(len(np.unique(y_test))))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score
import numpy as np

# 假设 y_test 是真实标签，y_pred 是模型预测的标签

# MAE
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# R-squared
# 注意：R-squared 通常用于回归问题。在分类问题中，它可能不适用。
# r2 = r2_score(y_test, y_pred)
# print("R-squared (R²):", r2)

# F-measure
f_measure = f1_score(y_test, y_pred, average='weighted')
print("F-measure:", f_measure)


# G-mean
def g_mean(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return np.prod(sensitivity) ** (1 / len(sensitivity))


gmean = g_mean(y_test, y_pred)
print("Geometric Mean (G-mean):", gmean)
