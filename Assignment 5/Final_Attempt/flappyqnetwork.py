import gymnasium as gym
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
from collections import deque


class FlappyQNetwork(nn.Module):
    def __init__(self, input_shape, actions):
        super(FlappyQNetwork,self).__init__()

        # The convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0],32,kernel_size=7,stride=3),
            nn.ReLU6(),
            nn.Conv2d(32,64,kernel_size=5,stride=2),
            nn.ReLU6(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU6(),
        )

        # The fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.get_conv_output_size(input_shape),256),
            nn.ReLU6(),
            nn.Linear(256,actions)
        )

    def get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1,*input_shape)
            output = self.conv(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0),-1))
        