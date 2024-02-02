import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, \
    f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from datasave import train_loader, test_loader
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


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


# 提取特征和标签
X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# 特征归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA 降维
# pca = PCA(n_components=0.95)  # 保留 95% 的方差
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

# 网格搜索参数进一步简化
parameters = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly']
}

# GridSearchCV
clf = GridSearchCV(SVC(), parameters, cv=2, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# 使用最佳参数的模型进行预测
model = clf.best_estimator_
y_pred = model.predict(X_test_scaled)

# 评估指标
accuracy = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)  # 注意: R2 一般用于回归任务
f_measure = f1_score(y_test, y_pred, average='weighted')


# G-mean 函数
def g_mean(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return np.prod(sensitivity) ** (1 / len(sensitivity))


gmean = g_mean(y_test, y_pred)

# 输出结果
print(f"Accuracy: {accuracy}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
print(f"F measure: {f_measure}")
print(f"G mean: {gmean}")

# 可视化混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
plt.imshow(conf_mat_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xticks(np.arange(len(np.unique(y_test))))
plt.yticks(np.arange(len(np.unique(y_test))))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
