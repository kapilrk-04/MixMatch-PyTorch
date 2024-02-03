'''
Implementation of the WeightEMA class for exponential moving average
'''

import numpy as np
import torch

class WeightEMA:
    def __init__(self, model, ema_model, alpha=0.999, weight_decay=0.02):
        # initialising required bookkeeping variables
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.weight_decay = weight_decay

    def step(self):
        # updating the exponential moving average of the model parameters
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                param.mul_(1 - self.weight_decay)

    def apply(self):
        # copying the parameters of the exponential moving average model to the original model
        for ema_param, param in zip(self.ema_params, self.params):
            param.data.copy_(ema_param.data)

