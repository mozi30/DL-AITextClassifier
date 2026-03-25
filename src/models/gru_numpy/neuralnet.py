#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import csv
import pickle
from pathlib import Path

import numpy as np

from src.models.gru_numpy.losses import MeanSquaredError
from src.models.gru_numpy.metrics import mse
from src.models.gru_numpy.optimizer import Optimizer
from src.dataloader import SentenceDataModule, SentenceDataLoader


ID_TO_VENDOR = {
    0: "Human",
    1: "OpenAI",
    2: "Google",
    3: "Anthropic",
    4: "Meta",
}


class NeuralNetwork:

    def __init__(self, datamodule: SentenceDataModule, epochs=100, batch_size=128, optimizer=None,
                 learning_rate=0.001, momentum=0.90, verbose=False,
                 loss=MeanSquaredError,
                 metric: callable = mse,
                 early_stopping_patience=5,
                 lr_patience=2,
                 lr_decay_factor=0.5,
                 min_learning_rate=1e-5,
                 restore_best_weights=True,
                 num_train_epoch_batches: int | None = None,
                 num_val_epoch_batches: int | None = None,
                 save_current_epoch_weights_path: str | None = None,
                 save_best_epoch_weights_path: str | None = None):
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
        self.num_train_epoch_batches = num_train_epoch_batches
        self.num_val_epoch_batches = num_val_epoch_batches
        self.save_current_epoch_weights_path = save_current_epoch_weights_path
        self.save_best_epoch_weights_path = save_best_epoch_weights_path

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
        batch_num = 0
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
        i = 1
        for X_batch, y_batch in self.get_dataload_mini_batch(loader, shuffle=False):
            if self.num_val_epoch_batches:
                print(f"Val Batch: {i} of {self.num_val_epoch_batches}", end="\r")
            else:
                print(f"Val Batch: {i} of {len(loader)/self.batch_size}", end="\r")
            i = i + 1
            outputs.append(self.forward_propagation(X_batch, training=False))
            labels.append(y_batch)
            if self.num_val_epoch_batches != None:
                if i >= self.num_val_epoch_batches:
                    break
        
            
            

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
            i = 1
            for X_batch, y_batch in self.get_dataload_mini_batch(self.train_loader):
                if self.num_train_epoch_batches:
                    print(f"Train Batch: {i} of {self.num_train_epoch_batches}", end="\r")
                else:
                    print(f"Train Batch: {i} of {len(self.train_loader)/self.batch_size}", end="\r")
                i = i + 1
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)
                batch_outputs.append(output)
                batch_labels.append(y_batch)
                if self.num_train_epoch_batches != None:
                    if i >= self.num_train_epoch_batches:
                        break
            print("\n")

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
                if self.save_best_epoch_weights_path:
                    self.save_weights(self.save_best_epoch_weights_path)
            else:
                epochs_without_improvement += 1
                lr_plateau_epochs += 1

            if self.save_current_epoch_weights_path:
                self.save_weights(self.save_current_epoch_weights_path)

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

    def predict_texts(self, texts, batch_size=None):
        if not texts:
            return np.array([])

        if batch_size is None:
            batch_size = self.batch_size or len(texts)

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            outputs.append(self.forward_propagation(batch, training=False))
        return np.concatenate(outputs)

    def predict_evaluation_subm2(self, input_csv: str = "datasets/evaluation/subm2.csv", output_csv: str | None = None,
                                  batch_size: int | None = None):
        """Run the GRU model on evaluation/subm2.csv and map outputs to vendor labels.

        The input CSV is expected to have columns "ID" and "Text" and to use
        ';' as a separator. Predictions are mapped using ID_TO_VENDOR, i.e.:
        0 -> Human, 1 -> OpenAI, 2 -> Google, 3 -> Anthropic, 4 -> Meta.

        If output_csv is provided, a submission-style CSV with columns
        ID,Label is written similar to subm2/subm2-g5-MEI-B.csv.
        """
        ids: list[str] = []
        texts: list[str] = []

        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                ids.append(row["ID"])
                texts.append(row["Text"])

        outputs = self.predict_texts(texts, batch_size=batch_size)
        if outputs.ndim == 1:
            pred_ids = (outputs > 0.5).astype(int)
        else:
            pred_ids = np.argmax(outputs, axis=1)

        labels = [ID_TO_VENDOR.get(int(idx), "Unknown") for idx in pred_ids]

        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Label"])
                writer.writerows(zip(ids, labels))

        return ids, labels

    def score(self, predictions=None, loader=None):
        target_loader = loader or self.test_loader
        labels = self._loader_labels(target_loader)
        if predictions is None:
            predictions = self.predict(loader=target_loader)
        if self.metric is not None:
            return self.metric(labels, predictions)
        raise ValueError("No metric specified for the neural network.")

    def save_weights(self, path: str):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        layer_states = []
        for layer in self.layers:
            layer_states.append({
                "layer_name": layer.layer_name(),
                "weights": layer.get_weights() if hasattr(layer, "get_weights") else {},
            })

        payload = {
            "layer_states": layer_states,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(payload, f)

    def load_weights(self, path: str, strict: bool = True):
        checkpoint_path = Path(path)
        with open(checkpoint_path, "rb") as f:
            payload = pickle.load(f)

        layer_states = payload.get("layer_states", [])

        if strict and len(layer_states) != len(self.layers):
            raise ValueError(
                f"Checkpoint has {len(layer_states)} layers, but model has {len(self.layers)} layers."
            )

        for idx, layer in enumerate(self.layers):
            if idx >= len(layer_states):
                break

            state = layer_states[idx]
            expected_name = layer.layer_name()
            saved_name = state.get("layer_name")
            if strict and saved_name != expected_name:
                raise ValueError(
                    f"Layer mismatch at index {idx}: checkpoint has {saved_name}, model has {expected_name}."
                )

            if hasattr(layer, "set_weights"):
                layer.set_weights(state.get("weights", {}))

        return self
