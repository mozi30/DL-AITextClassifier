#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Optimizer:

    def __init__(self, learning_rate = 0.001, momentum = 0.90, beta2 = 0.999,
                 epsilon = 1e-8, method = "adam", clipnorm = 1.0, weight_decay = 0.0):
        self.first_moment = None
        self.second_moment = None
        self.timestep = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta2 = beta2
        self.epsilon = epsilon
        self.method = method
        self.clipnorm = clipnorm
        self.weight_decay = weight_decay

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def _clip_gradient(self, gradient):
        if self.clipnorm is None:
            return gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm == 0 or grad_norm <= self.clipnorm:
            return gradient
        return gradient * (self.clipnorm / (grad_norm + 1e-12))
 
    def update(self, w, grad_loss_w):
        gradient = self._clip_gradient(grad_loss_w)
        if self.weight_decay != 0.0:
            gradient = gradient + self.weight_decay * w

        if self.method == "sgd":
            if self.first_moment is None:
                self.first_moment = np.zeros(np.shape(w))
            self.first_moment = self.momentum * self.first_moment + (1 - self.momentum) * gradient
            return w - self.learning_rate * self.first_moment

        if self.first_moment is None:
            self.first_moment = np.zeros(np.shape(w))
            self.second_moment = np.zeros(np.shape(w))

        self.timestep += 1
        self.first_moment = self.momentum * self.first_moment + (1 - self.momentum) * gradient
        self.second_moment = self.beta2 * self.second_moment + (1 - self.beta2) * (gradient ** 2)

        first_unbiased = self.first_moment / (1 - self.momentum ** self.timestep)
        second_unbiased = self.second_moment / (1 - self.beta2 ** self.timestep)

        return w - self.learning_rate * first_unbiased / (np.sqrt(second_unbiased) + self.epsilon)
