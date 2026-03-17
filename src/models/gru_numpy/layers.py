#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import copy

import numpy as np

from src.text_embedding import BaseEmbedding


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__

    def set_learning_rate(self, learning_rate):
        return None


class EmbeddingLayer(Layer):
    def __init__(self, embedding: BaseEmbedding):
        super().__init__()
        self.embedding = embedding
        self.embedding_cache = []
        self.sequence_lengths = []
        self.w_opt = None

    def initialize(self, optimizer):
        self.w_opt = copy.deepcopy(optimizer)
        return self

    def forward_propagation(self, inputs, training=True):
        self.embedding_cache = []
        self.sequence_lengths = []
        batch_embeddings = []
        max_seq_len = 1

        for sentence in inputs:
            tokens, embeddings = self.embedding.process_sentence(sentence)
            batch_embeddings.append(embeddings)
            self.embedding_cache.append(tokens)
            self.sequence_lengths.append(len(tokens))
            if embeddings.shape[0] > 0:
                max_seq_len = max(max_seq_len, embeddings.shape[0])

        output = np.zeros((len(batch_embeddings), max_seq_len, self.embedding.embedding_size), dtype=float)
        for index, embeddings in enumerate(batch_embeddings):
            if embeddings.shape[0] > 0:
                output[index, :embeddings.shape[0], :] = embeddings

        self.output = output
        return self.output

    def backward_propagation(self, error):
        if self.w_opt is None or error is None:
            return error

        learning_rate = self.w_opt.learning_rate
        for sample_index, tokens in enumerate(self.embedding_cache):
            for token_index, token in enumerate(tokens):
                self.embedding.update_embedding(token, error[sample_index, token_index], lr=learning_rate)

        return error

    def output_shape(self):
        return (None, self.embedding.embedding_size)

    def parameters(self):
        return int(np.prod(self.embedding.encoded.shape))

    def set_learning_rate(self, learning_rate):
        if self.w_opt is not None:
            self.w_opt.set_learning_rate(learning_rate)


class GRULayer(Layer):
    def __init__(self, n_units=None, input_shape=None, return_sequences=False, seed=42,
                 batch_size=None, num_input=None, num_hidden=None):
        super().__init__()
        if n_units is None:
            n_units = num_hidden
        if n_units is None:
            raise ValueError("GRULayer requires n_units or num_hidden.")
        if input_shape is None and num_input is not None:
            input_shape = (None, num_input)

        self.n_units = n_units
        self._input_shape = input_shape
        self.return_sequences = return_sequences
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.input = None
        self.output = None

        self.seq_len = None
        self.input_dim = None
        self.sequence_mask = None
        self.sequence_lengths = None

        self.Wxr = None
        self.Whr = None
        self.br = None

        self.Wxz = None
        self.Whz = None
        self.bz = None

        self.Wxh = None
        self.Whh = None
        self.bh = None

        self.wxr_opt = None
        self.whr_opt = None
        self.br_opt = None

        self.wxz_opt = None
        self.whz_opt = None
        self.bz_opt = None

        self.wxh_opt = None
        self.whh_opt = None
        self.bh_opt = None

        self.step_cache = []
        self._initialized = False

    def _glorot_uniform(self, fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, (fan_in, fan_out))

    def _orthogonal(self, size):
        matrix = self.rng.standard_normal((size, size))
        q, _ = np.linalg.qr(matrix)
        return q

    def _initialize_parameters(self):
        if self.input_dim is None or self._initialized:
            return self

        self.Wxr = self._glorot_uniform(self.input_dim, self.n_units)
        self.Whr = self._orthogonal(self.n_units)
        self.br = np.zeros((1, self.n_units))

        self.Wxz = self._glorot_uniform(self.input_dim, self.n_units)
        self.Whz = self._orthogonal(self.n_units)
        self.bz = np.zeros((1, self.n_units))

        self.Wxh = self._glorot_uniform(self.input_dim, self.n_units)
        self.Whh = self._orthogonal(self.n_units)
        self.bh = np.zeros((1, self.n_units))

        self._initialized = True
        return self

    def initialize(self, optimizer):
        self.wxr_opt = copy.deepcopy(optimizer)
        self.whr_opt = copy.deepcopy(optimizer)
        self.br_opt = copy.deepcopy(optimizer)

        self.wxz_opt = copy.deepcopy(optimizer)
        self.whz_opt = copy.deepcopy(optimizer)
        self.bz_opt = copy.deepcopy(optimizer)

        self.wxh_opt = copy.deepcopy(optimizer)
        self.whh_opt = copy.deepcopy(optimizer)
        self.bh_opt = copy.deepcopy(optimizer)

        if self.input_shape() is not None:
            self.seq_len, self.input_dim = self.input_shape()

        self._initialize_parameters()
        return self

    def parameters(self):
        total = 0
        for parameter in [
            self.Wxr, self.Whr, self.br,
            self.Wxz, self.Whz, self.bz,
            self.Wxh, self.Whh, self.bh,
        ]:
            if parameter is not None:
                total += int(np.prod(parameter.shape))
        return total

    def output_shape(self):
        if self.return_sequences:
            return (self.seq_len, self.n_units)
        return (self.n_units,)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_backward(self, y):
        return y * (1.0 - y)

    def _infer_sequence_mask(self, inputs):
        return (np.linalg.norm(inputs, axis=2) > 0).astype(float)

    def _gather_last_valid_outputs(self, outputs):
        batch_size = outputs.shape[0]
        last_indices = np.clip(self.sequence_lengths - 1, 0, outputs.shape[1] - 1)
        return outputs[np.arange(batch_size), last_indices, :]

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        batch_size, seq_len, input_dim = inputs.shape

        self.seq_len = seq_len
        if self.input_dim is None:
            self.input_dim = input_dim
        if not self._initialized:
            self._initialize_parameters()
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {input_dim}.")

        self.sequence_mask = self._infer_sequence_mask(inputs)
        self.sequence_lengths = self.sequence_mask.sum(axis=1).astype(int)

        h_prev = np.zeros((batch_size, self.n_units))
        outputs = []
        self.step_cache = []

        for timestep in range(seq_len):
            x_t = inputs[:, timestep, :]
            mask_t = self.sequence_mask[:, timestep][:, None]

            r_t = self._sigmoid(x_t @ self.Wxr + h_prev @ self.Whr + self.br)
            z_t = self._sigmoid(x_t @ self.Wxz + h_prev @ self.Whz + self.bz)
            h_tilde = np.tanh(x_t @ self.Wxh + (r_t * h_prev) @ self.Whh + self.bh)
            h_candidate = (1.0 - z_t) * h_prev + z_t * h_tilde
            h_t = mask_t * h_candidate + (1.0 - mask_t) * h_prev

            if training:
                self.step_cache.append({
                    "x": x_t,
                    "h_prev": h_prev,
                    "r": r_t,
                    "z": z_t,
                    "h_tilde": h_tilde,
                    "mask": mask_t,
                })

            outputs.append(mask_t * h_t)
            h_prev = h_t

        outputs = np.stack(outputs, axis=1)

        if self.return_sequences:
            self.output = outputs
        else:
            self.output = self._gather_last_valid_outputs(outputs)

        return self.output

    def backward_propagation(self, output_error):
        batch_size, seq_len, _ = self.input.shape

        dWxr = np.zeros_like(self.Wxr)
        dWhr = np.zeros_like(self.Whr)
        dbr = np.zeros_like(self.br)

        dWxz = np.zeros_like(self.Wxz)
        dWhz = np.zeros_like(self.Whz)
        dbz = np.zeros_like(self.bz)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        input_error = np.zeros_like(self.input)
        dh_next = np.zeros((batch_size, self.n_units))

        if self.return_sequences:
            doutputs = output_error
        else:
            doutputs = np.zeros((batch_size, seq_len, self.n_units))
            last_indices = np.clip(self.sequence_lengths - 1, 0, seq_len - 1)
            doutputs[np.arange(batch_size), last_indices, :] = output_error

        for timestep in reversed(range(seq_len)):
            cache = self.step_cache[timestep]

            x = cache["x"]
            h_prev = cache["h_prev"]
            r = cache["r"]
            z = cache["z"]
            h_tilde = cache["h_tilde"]
            mask = cache["mask"]

            dh = doutputs[:, timestep, :] + dh_next
            dh_valid = dh * mask
            dh_prev = dh * (1.0 - mask)

            dh_tilde = dh_valid * z
            dz = dh_valid * (h_tilde - h_prev)
            dh_prev += dh_valid * (1.0 - z)

            da_h = dh_tilde * (1.0 - h_tilde ** 2)

            dWxh += x.T @ da_h
            dWhh += (r * h_prev).T @ da_h
            dbh += np.sum(da_h, axis=0, keepdims=True)

            dx_h = da_h @ self.Wxh.T
            d_rh_prev = da_h @ self.Whh.T

            dr = d_rh_prev * h_prev
            dh_prev += d_rh_prev * r

            da_r = dr * self._sigmoid_backward(r)

            dWxr += x.T @ da_r
            dWhr += h_prev.T @ da_r
            dbr += np.sum(da_r, axis=0, keepdims=True)

            dx_r = da_r @ self.Wxr.T
            dh_prev += da_r @ self.Whr.T

            da_z = dz * self._sigmoid_backward(z)

            dWxz += x.T @ da_z
            dWhz += h_prev.T @ da_z
            dbz += np.sum(da_z, axis=0, keepdims=True)

            dx_z = da_z @ self.Wxz.T
            dh_prev += da_z @ self.Whz.T

            input_error[:, timestep, :] = (dx_h + dx_r + dx_z) * mask
            dh_next = dh_prev

        self.Wxr = self.wxr_opt.update(self.Wxr, dWxr)
        self.Whr = self.whr_opt.update(self.Whr, dWhr)
        self.br = self.br_opt.update(self.br, dbr)

        self.Wxz = self.wxz_opt.update(self.Wxz, dWxz)
        self.Whz = self.whz_opt.update(self.Whz, dWhz)
        self.bz = self.bz_opt.update(self.bz, dbz)

        self.Wxh = self.wxh_opt.update(self.Wxh, dWxh)
        self.Whh = self.whh_opt.update(self.Whh, dWhh)
        self.bh = self.bh_opt.update(self.bh, dbh)

        return input_error

    def set_learning_rate(self, learning_rate):
        for optimizer in [
            self.wxr_opt, self.whr_opt, self.br_opt,
            self.wxz_opt, self.whz_opt, self.bz_opt,
            self.wxh_opt, self.whh_opt, self.bh_opt,
        ]:
            if optimizer is not None:
                optimizer.set_learning_rate(learning_rate)


class BiGRULayer(Layer):
    def __init__(self, n_units=None, input_shape=None, return_sequences=False, seed=42,
                 batch_size=None, num_input=None, num_hidden=None):
        super().__init__()
        if n_units is None:
            n_units = num_hidden
        if n_units is None:
            raise ValueError("BiGRULayer requires n_units or num_hidden.")

        inferred_input_shape = input_shape if input_shape is not None else (None, num_input) if num_input is not None else None
        self.n_units = n_units
        self._input_shape = inferred_input_shape
        self.return_sequences = return_sequences
        self.forward_gru = GRULayer(
            n_units=n_units,
            input_shape=inferred_input_shape,
            return_sequences=return_sequences,
            seed=seed,
            batch_size=batch_size,
        )
        self.backward_gru = GRULayer(
            n_units=n_units,
            input_shape=inferred_input_shape,
            return_sequences=return_sequences,
            seed=seed + 1,
            batch_size=batch_size,
        )

    def set_input_shape(self, input_shape):
        super().set_input_shape(input_shape)
        self.forward_gru.set_input_shape(input_shape)
        self.backward_gru.set_input_shape(input_shape)

    def initialize(self, optimizer):
        self.forward_gru.initialize(optimizer)
        self.backward_gru.initialize(optimizer)
        return self

    def forward_propagation(self, inputs, training=True):
        forward_output = self.forward_gru.forward_propagation(inputs, training)
        backward_output = self.backward_gru.forward_propagation(inputs[:, ::-1, :], training)

        if self.return_sequences:
            backward_output = backward_output[:, ::-1, :]

        self.output = np.concatenate([forward_output, backward_output], axis=-1)
        return self.output

    def backward_propagation(self, output_error):
        if self.return_sequences:
            forward_error = output_error[:, :, :self.n_units]
            backward_error = output_error[:, :, self.n_units:][:, ::-1, :]
            grad_forward = self.forward_gru.backward_propagation(forward_error)
            grad_backward = self.backward_gru.backward_propagation(backward_error)
            return grad_forward + grad_backward[:, ::-1, :]

        forward_error = output_error[:, :self.n_units]
        backward_error = output_error[:, self.n_units:]
        grad_forward = self.forward_gru.backward_propagation(forward_error)
        grad_backward = self.backward_gru.backward_propagation(backward_error)
        return grad_forward + grad_backward[:, ::-1, :]

    def output_shape(self):
        if self.return_sequences:
            return (self.forward_gru.seq_len, self.n_units * 2)
        return (self.n_units * 2,)

    def parameters(self):
        return self.forward_gru.parameters() + self.backward_gru.parameters()

    def set_learning_rate(self, learning_rate):
        self.forward_gru.set_learning_rate(learning_rate)
        self.backward_gru.set_learning_rate(learning_rate)


class SequenceMeanMaxPool(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None
        self.valid_counts = None
        self.max_selector = None
        self.max_counts = None
        self.feature_dim = None

    def forward_propagation(self, inputs, training=True):
        self.feature_dim = inputs.shape[2]
        self.mask = (np.linalg.norm(inputs, axis=2, keepdims=True) > 0).astype(float)
        self.valid_counts = np.clip(self.mask.sum(axis=1), 1.0, None)

        mean_features = np.sum(inputs * self.mask, axis=1) / self.valid_counts

        masked_inputs = np.where(self.mask.astype(bool), inputs, -np.inf)
        max_features = np.max(masked_inputs, axis=1)
        max_features[~np.isfinite(max_features)] = 0.0

        self.max_selector = ((inputs == max_features[:, None, :]).astype(float)) * self.mask
        self.max_counts = np.clip(self.max_selector.sum(axis=1), 1.0, None)

        self.output = np.concatenate([mean_features, max_features], axis=1)
        return self.output

    def backward_propagation(self, output_error):
        mean_error = output_error[:, :self.feature_dim]
        max_error = output_error[:, self.feature_dim:]

        mean_grad = (mean_error[:, None, :] * self.mask) / self.valid_counts[:, None, :]
        max_grad = (max_error[:, None, :] * self.max_selector) / self.max_counts[:, None, :]
        return mean_grad + max_grad

    def output_shape(self):
        return (self._input_shape[1] * 2,)

    def parameters(self):
        return 0


class DropoutLayer(Layer):
    def __init__(self, rate=0.5, seed=42):
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0, 1).")
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward_propagation(self, inputs, training=True):
        if not training or self.rate == 0.0:
            self.mask = None
            self.output = inputs
            return self.output

        self.mask = (self.rng.random(inputs.shape) < self.keep_prob).astype(float) / self.keep_prob
        self.output = inputs * self.mask
        return self.output

    def backward_propagation(self, output_error):
        if self.mask is None:
            return output_error
        return output_error * self.mask

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0


class DenseLayer(Layer):
    def __init__(self, n_units, input_shape=None, seed=42):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.rng = np.random.default_rng(seed)

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.w_opt = None
        self.b_opt = None

    def initialize(self, optimizer):
        fan_in = self.input_shape()[0]
        limit = np.sqrt(6.0 / (fan_in + self.n_units))
        self.weights = self.rng.uniform(-limit, limit, (fan_in, self.n_units))
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return int(np.prod(self.weights.shape) + np.prod(self.biases.shape))

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error

    def output_shape(self):
        return (self.n_units,)

    def set_learning_rate(self, learning_rate):
        if self.w_opt is not None:
            self.w_opt.set_learning_rate(learning_rate)
        if self.b_opt is not None:
            self.b_opt.set_learning_rate(learning_rate)
