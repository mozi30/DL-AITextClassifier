#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
from src.models.gru_numpy.layers import Layer

class ActivationLayer(Layer):

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0
    
class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        sig = self.activation_function(input)
        return sig * (1 - sig)


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)
    

class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input):
        # numerically stable softmax
        shifted = input - np.max(input, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def derivative(self, input):
        # derivative assumes input already passed through softmax
        s = self.activation_function(input)
        return s * (1 - s)

    def backward_propagation(self, output_error):
        return output_error
    

    
