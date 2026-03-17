#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

class LossFunction:

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    
    def loss(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true, y_pred):
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (p - y_true) / (p * (1 - p))
    
class CrossEntropyLoss(LossFunction):

    def _prepare_targets(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        if y_true.ndim == 1:
            return np.eye(y_pred.shape[1], dtype=float)[y_true.astype(int)]
        return y_true

    def loss(self, y_true, y_pred):
        eps = 1e-12
        y_true = self._prepare_targets(y_true, y_pred)
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # average over batch
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true, y_pred):
        y_true = self._prepare_targets(y_true, y_pred)
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size
    

    
    
