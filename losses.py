'''
Implementation of the MixMatch loss function
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixMatchLoss(nn.Module):
    def __init__(self, alpha, lambda_u, num_classes, temperature, total_epochs):
        super(MixMatchLoss, self).__init__()
        self.alpha = alpha
        self.lambda_u = lambda_u
        self.num_classes = num_classes
        self.temperature = temperature
        self.total_epochs = total_epochs

    def forward(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        # print(len(outputs_x), len(targets_x), len(outputs_u), len(targets_u))

        # Calculate supervised loss (Lx)
        supervised_loss = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

        # Calculate unsupervised loss (Lu)
        probs_u = torch.softmax(outputs_u, dim=1)
        unsupervised_loss = torch.mean((probs_u - targets_u) ** 2)

        return supervised_loss + self.lambda_u * float(np.clip(epoch / self.total_epochs, 0.0, 1.0)) * unsupervised_loss

        '''
        Used for evaluating and observing each compoenent of the loss function individually;
        uncomment the following line and comment the above line to get the individual components of the loss function
        Below line is used for experimentation and observation, along with building the graphs
        '''
        # return supervised_loss, unsupervised_loss, self.lambda_u * float(np.clip(epoch / self.total_epochs, 0.0, 1.0))

