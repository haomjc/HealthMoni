# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import re
import scipy.io as scio
import scipy.signal
from torch.utils import data as da
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py
import tables

raw_num = 100

time_series_length = 4096


class Data(object):

    def __init__(self):
        self.data = self.get_data()
        self.label = self.get_label()

    def file_list(self):
        return os.listdir('../data/')

    def get_data(self):
        file_list = self.file_list()
        x = np.empty((time_series_length, 0))
        for i in range(len(file_list)):
            # file = scio.loadmat('./data/{}'.format(file_list[i]))
            # file = scio.loadmat('./data/{}'.format(file_list[i+1]))
            # file = h5py.File('./data/{}'.format(file_list[i+1]),"r")

            file = tables.open_file('../data/{}'.format(file_list[i]), mode="r")

            print(i)

            for k in file.root:
                # print(k, "k")
                file_matched = re.match('Data', k._v_name)

                if file_matched:
                    key = file_matched.group()
                    # print(key)
            # data1 = np.array(file[key][0:102400])  # 0:80624
            # data1 = np.array(getattr(k.read(), 'Data')[0:102400])
            data1 = np.array(k.read())[:, :(time_series_length * raw_num)].T
            # data1 = np.expand_dims(data1, axis=1)

            # print(data1.shape)

            for j in range(0, len(data1) - (time_series_length - 1), time_series_length):
                # print(data1[j:j + 1024].shape)

                x = np.concatenate((x, data1[j:j + time_series_length, :]), axis=1)

                # print(x.shape)

            file.close()
        return x.T

    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.mat', '') for i in file_list])
        label = title[:, np.newaxis]
        label_copy = np.copy(label)
        for _ in range(raw_num - 1):
            label = np.hstack((label, label_copy))

        # print(label.shape)

        return label.flatten()


Data = Data()
data = Data.data
label = Data.label
y = label.astype("int32")

ss = MinMaxScaler()
data = data.T

# print(data.shape)
# print(y.shape)
# print(data)
data = ss.fit_transform(data).T
data = data.reshape(2800, 4, time_series_length)
# print(data.shape, "data")

# print(y.shape, "y")

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=2, stratify=y)
# X_train = torch.from_numpy(X_train).unsqueeze(2)
X_train = torch.from_numpy(X_train)

# print(X_train.shape, "X_train")
# X_test = torch.from_numpy(X_test).unsqueeze(2)
X_test = torch.from_numpy(X_test)


# print(X_test.shape, "X_test")


class TrainDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train
        self.Label = y_train

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label

    def __len__(self):
        return len(self.Data)


class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_test
        self.Label = y_test

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label

    def __len__(self):
        return len(self.Data)


Train = TrainDataset()
Test = TestDataset()
train_loader = da.DataLoader(Train, batch_size=128, shuffle=True)
test_loader = da.DataLoader(Test, batch_size=10, shuffle=False)
