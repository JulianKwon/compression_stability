from __future__ import print_function, division

import os
# Ignore warnings
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import save_data

warnings.filterwarnings("ignore")


import torch
from torch.utils import data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_loader(x, y, batch_size, shuffle=True, drop_last=True):
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    #     print(x.shape, y.shape)
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return loader


class MetroTrafficDataset:
    def __init__(self, file, root_dir, window_size=24 * 7, prediction_length=24, test_size=0.2, transform=None):
        super().__init__()
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.root_dir = root_dir
        self.file = file
        self.data, self.columns, self.scaler = self._preprocessing()
        self.transform = transform
        self.test_size = test_size
        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test = self._window_shifting()

    def make_dataloader(self, type='train', batch_size=1000):
        if type == 'train':
            loader = get_loader(self.X_train.reshape(self.X_train.shape[0], -1), self.y_train, batch_size)
        elif type == 'valid':
            loader = get_loader(self.X_valid.reshape(self.X_valid.shape[0], -1), self.y_valid, batch_size)
        else:
            loader = get_loader(self.X_test.reshape(self.X_test.shape[0], -1), self.y_test, batch_size)

        return loader

    def __channel__(self):
        return self.data.shape[1]

    def __len__(self):
        """

        :return: data_size
        """
        return len(self.data)

    def save_scaler(self):
        save_data(self.scaler, '../data_', 'scaler.pkl')

    def _preprocessing(self):
        df = pd.read_csv(os.path.join(self.root_dir, self.file))

        holiday_one_hot = pd.get_dummies(df.holiday)
        weather_main_one_hot = pd.get_dummies(df.weather_main)
        weather_description_one_hot = pd.get_dummies(df.weather_description)

        numeric_data = df.drop(['holiday', 'weather_main', 'weather_description', 'date_time'], axis=1)

        numeric_columns = numeric_data.columns
        columns = np.concatenate((numeric_data.columns, holiday_one_hot.columns, weather_main_one_hot.columns,
                                  weather_description_one_hot.columns))

        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(numeric_data)

        data = np.concatenate((numeric_data, holiday_one_hot, weather_main_one_hot, weather_description_one_hot),
                              axis=1)

        return data, columns, scaler

    def _window_shifting(self):
        """

        :return:
             X_total_train,
             y_total_train,
             X_total_valid,
             y_total_valid,
             X_total_test,
             y_total_test
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.data[:, 4], test_size=self.test_size,
                                                            shuffle=False)

        validation_spot = int(len(X_train) * 0.8)
        test_perm = np.random.permutation(len(X_test) - (self.window_size + self.prediction_length) + 1)

        X_total_test = np.zeros(
            (X_test.shape[0] - (self.window_size + self.prediction_length) + 1, X_test.shape[1], self.window_size))
        y_total_test = np.zeros(
            (X_test.shape[0] - (self.window_size + self.prediction_length) + 1, self.prediction_length))

        X_total_test, y_total_test = self._make_dataset(X_total_test, y_total_test, X_test, test_perm)

        _X_train = X_train[:validation_spot]
        _X_valid = X_train[validation_spot:]
        _y_train = y_train[:validation_spot]
        _y_valid = y_train[validation_spot:]

        perm_train = np.random.permutation(len(_X_train) - (self.window_size + self.prediction_length) + 1)
        perm_valid = np.random.permutation(len(_X_valid) - (self.window_size + self.prediction_length) + 1)

        X_total_train = np.zeros(
            (len(_X_train) - (self.window_size + self.prediction_length) + 1, _X_train.shape[1], self.window_size))
        y_total_train = np.zeros(
            (len(_y_train) - (self.window_size + self.prediction_length) + 1, self.prediction_length))
        X_total_valid = np.zeros(
            (len(_X_valid) - (self.window_size + self.prediction_length) + 1, _X_train.shape[1], self.window_size))
        y_total_valid = np.zeros(
            (len(_y_valid) - (self.window_size + self.prediction_length) + 1, self.prediction_length))

        X_total_train, y_total_train = self._make_dataset(X_total_train, y_total_train, _X_train, perm_train)
        X_total_valid, y_total_valid = self._make_dataset(X_total_valid, y_total_valid, _X_valid, perm_valid)

        return X_total_train, y_total_train, X_total_valid, y_total_valid, X_total_test, y_total_test

    def _make_dataset(self, X_total, y_total, data, perm):
        for j, idx in enumerate(perm):
            X = data[idx:idx + self.window_size]
            y = data[idx + self.window_size:idx + self.window_size + self.prediction_length][:, 4]
            X = np.transpose(X)
            X_total[j] = X
            y_total[j] = y

        return X_total, y_total
