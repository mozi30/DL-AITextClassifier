#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_pred.ndim > 1:
        if y_pred.shape[1] == 1:
            y_pred = np.round(y_pred[:, 0])
        else:
            y_pred = np.argmax(y_pred, axis=1)

    if y_true.ndim > 1:
        if y_true.shape[1] == 1:
            y_true = y_true[:, 0]
        else:
            y_true = np.argmax(y_true, axis=1)

    return np.mean(y_pred == y_true)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)
