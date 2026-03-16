#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np

from src.models.gru_numpy.losses import MeanSquaredError
from src.models.gru_numpy.metrics import mse
from src.models.gru_numpy.optimizer import Optimizer
from src.dataloader import SentenceDataModule, SentenceDataLoader


class NeuralNetwork:

    def __init__(self, datamodule: SentenceDataModule, epochs=100, batch_size=128, optimizer=None,
                 learning_rate=0.001, momentum=0.90, verbose=False,
                 loss=MeanSquaredError,
                 metric: callable = mse,
                 early_stopping_patience=5,
                 lr_patience=2,
                 lr_decay_factor=0.5,
                 min_learning_rate=1e-5,
                 restore_best_weights=True):
        self.train_loader = datamodule.get_train_loader()
        self.val_loader = datamodule.get_val_loader()
        self.test_loader = datamodule.get_test_loader()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer or Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric
        self.early_stopping_patience = early_stopping_patience
        self.lr_patience = lr_patience
        self.lr_decay_factor = lr_decay_factor
        self.min_learning_rate = min_learning_rate
        self.restore_best_weights = restore_best_weights

        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, "initialize"):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if y is not None:
                yield X[batch_indices], y[batch_indices]
            else:
                yield X[batch_indices], None

    def get_dataload_mini_batch(self, loader: SentenceDataLoader, shuffle=True):
        n_samples = len(loader)
        indices = np.arange(n_samples)
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n_samples, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_samples = [loader[int(index)] for index in batch_indices]
            batch_text = [sample["text"] for sample in batch_samples]
            batch_labels = np.array([sample["model"] for sample in batch_samples], dtype=int)
            yield batch_text, batch_labels

    def _loader_labels(self, loader: SentenceDataLoader):
        return np.array([loader[index]["model"] for index in range(len(loader))], dtype=int)

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def _evaluate_loader(self, loader: SentenceDataLoader):
        outputs = []
        labels = []
        for X_batch, y_batch in self.get_dataload_mini_batch(loader, shuffle=False):
            outputs.append(self.forward_propagation(X_batch, training=False))
            labels.append(y_batch)

        if not outputs:
            return None, None

        y_all = np.concatenate(labels)
        output_all = np.concatenate(outputs)
        loss = self.loss.loss(y_all, output_all)
        metric = self.metric(y_all, output_all) if self.metric is not None else None
        return loss, metric

    def _set_learning_rate(self, learning_rate):
        self.optimizer.set_learning_rate(learning_rate)
        for layer in self.layers:
            if hasattr(layer, "set_learning_rate"):
                layer.set_learning_rate(learning_rate)

    def fit(self):
        self.history = {}
        best_val_loss = np.inf
        best_layers = None
        epochs_without_improvement = 0
        lr_plateau_epochs = 0

        for epoch in range(1, self.epochs + 1):
            batch_outputs = []
            batch_labels = []

            for X_batch, y_batch in self.get_dataload_mini_batch(self.train_loader):
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)

                batch_outputs.append(output)
                batch_labels.append(y_batch)

            output_x_all = np.concatenate(batch_outputs)
            y_all = np.concatenate(batch_labels)

            train_loss = self.loss.loss(y_all, output_x_all)
            train_metric = self.metric(y_all, output_x_all) if self.metric is not None else None
            val_loss, val_metric = self._evaluate_loader(self.val_loader)

            self.history[epoch] = {
                "train_loss": train_loss,
                "train_metric": train_metric,
                "val_loss": val_loss,
                "val_metric": val_metric,
                "learning_rate": self.optimizer.learning_rate,
            }

            improved = val_loss is not None and val_loss < best_val_loss - 1e-6
            if improved:
                best_val_loss = val_loss
                best_layers = copy.deepcopy(self.layers)
                epochs_without_improvement = 0
                lr_plateau_epochs = 0
            else:
                epochs_without_improvement += 1
                lr_plateau_epochs += 1

            if self.lr_patience is not None and lr_plateau_epochs >= self.lr_patience:
                new_lr = max(self.optimizer.learning_rate * self.lr_decay_factor, self.min_learning_rate)
                if new_lr < self.optimizer.learning_rate:
                    self._set_learning_rate(new_lr)
                lr_plateau_epochs = 0

            if self.verbose:
                train_metric_text = f"{self.metric.__name__}: {train_metric:.4f}" if train_metric is not None else "metric: NA"
                val_metric_text = f"val_{self.metric.__name__}: {val_metric:.4f}" if val_metric is not None else "val_metric: NA"
                print(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"train_loss: {train_loss:.4f} - {train_metric_text} - "
                    f"val_loss: {val_loss:.4f} - {val_metric_text} - "
                    f"lr: {self.optimizer.learning_rate:.6f}"
                )

            if self.early_stopping_patience is not None and epochs_without_improvement >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        if self.restore_best_weights and best_layers is not None:
            self.layers = best_layers

        return self

    def predict(self, loader=None):
        target_loader = loader or self.test_loader
        predictions = []
        for texts, _ in self.get_dataload_mini_batch(target_loader, shuffle=False):
            predictions.append(self.forward_propagation(texts, training=False))
        return np.concatenate(predictions)

    def score(self, predictions=None, loader=None):
        target_loader = loader or self.test_loader
        labels = self._loader_labels(target_loader)
        if predictions is None:
            predictions = self.predict(loader=target_loader)
        if self.metric is not None:
            return self.metric(labels, predictions)
        raise ValueError("No metric specified for the neural network.")
