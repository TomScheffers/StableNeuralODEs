# Inspired by:
# https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
# https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.training.tensor_ops import Flatten

class GradientNet(nn.Module):
    def __init__(self, channels):
        super(GradientNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, stride=1, padding=1),
            nn.GroupNorm(channels, channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, stride=1, padding=1),
            nn.GroupNorm(channels, channels),
            nn.ReLU(inplace=True)
        )

    def concat_time(self, t, x):
        return torch.cat([x, torch.ones_like(x[:, :1, :, :]) * t], 1)

    def forward(self, t, x):
        x = self.block1(self.concat_time(t, x))
        return self.block2(self.concat_time(t, x))

class OdeNet(nn.Module):
    def __init__(self, solver, integrator, t0, t1, steps, in_channels, channels, classes):
        super(OdeNet, self).__init__()

        # First we want to downsample the image to have an acceptable size for the ODE solver (stride of 2 causes reduction of H/W)
        self.sampling = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=1),    # [batch_size, 1, 28/32, 28/32] -> [batch_size, channels, 28/32, 28/32]
            nn.GroupNorm(channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),  # [batch_size, channels, 28/32, 28/32] -> [batch_size, channels, 13/14, 13/14]
            nn.GroupNorm(channels, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),  # [batch_size, channels, 13/14, 13/14] -> [batch_size, channels, 6/7, 6/7]
            nn.GroupNorm(channels, channels),
            nn.ReLU(inplace=True)
        )  
        
        # Then we want to apply the magic: model differences so we can use ode solvers
        self.gradient = GradientNet(channels=channels)
        self.solver = solver(self.gradient, t0, t1, integrator, steps=1)

        # To make predictions we use a fully connected layer (so convenient :))
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(channels, classes)
        )

        # Define an optimizer so we can call it from the outside
        params = list(self.sampling.parameters()) + list(self.gradient.parameters()) + list(self.output.parameters())
        self.optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
        #self.optimizer = torch.optim.Adadelta(params)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.loss_module = nn.CrossEntropyLoss()

        # Check if GPU is available
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.sampling = torch.nn.DataParallel(self.sampling)
            self.gradient = torch.nn.DataParallel(self.gradient)
            self.cuda()
            self.device = 'cuda'

        # Print the amount of parameters
        print("This model is using %d parameters" % (sum(p.numel() for p in params if p.requires_grad)))

    def forward(self, x):
        x = self.sampling(x)
        x = self.solver.integrate(x)
        return self.output(x)
        