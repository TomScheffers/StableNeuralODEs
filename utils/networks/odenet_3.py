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
			nn.ReLU(inplace=True),
			nn.Conv2d(channels + 1, channels, 3, stride=1, padding=1),
			nn.BatchNorm2d(channels),
		)
		self.block2 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(channels + 1, channels, 3, stride=1, padding=1),
			nn.BatchNorm2d(channels),
		)

	def concat_time(self, t, x):
		return torch.cat([x, torch.ones_like(x[:, :1, :, :]) * t], 1)

	def forward(self, t, x):
		x = self.block1(self.concat_time(t, x))
		return self.block2(self.concat_time(t, x))

class OdeNet(nn.Module):
	def __init__(self, solver, integrator, t0, t1, steps, in_channels, channels, classes, mu=None, std=None):
		super(OdeNet, self).__init__()
		# We have 4 blocks of downsampling and ode solving
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, channels[0], 3, stride=1, padding=1),    # [batch_size, 1, 32, 32] -> [batch_size, channels[0], 32, 32]
			nn.BatchNorm2d(channels[0]),
		)
		self.gradient1 = GradientNet(channels=channels[0])
		self.solver1 = solver(self.gradient1, t0, t1, integrator, steps=steps, verbose=True, downsample=self.conv1)

		self.conv2 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),    # [batch_size, channels[0], 32, 32] -> [batch_size, channels[1], 16, 16]
			nn.BatchNorm2d(channels[1]),
		)
		self.gradient2 = GradientNet(channels=channels[1])
		self.solver2 = solver(self.gradient2, t0, t1, integrator, steps=steps, downsample=self.conv2)

		self.conv3 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),    # [batch_size, channels[1], 16, 16] -> [batch_size, channels[2], 8, 8]
			nn.BatchNorm2d(channels[2]),
		)
		self.gradient3 = GradientNet(channels=channels[2])
		self.solver3 = solver(self.gradient3, t0, t1, integrator, steps=steps, downsample=self.conv3)

		self.conv4 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1),    # [batch_size, channels[2], 8, 8] -> [batch_size, channels[3], 4, 4]
			nn.BatchNorm2d(channels[3]),
		)
		self.gradient4 = GradientNet(channels=channels[3])
		self.solver4 = solver(self.gradient4, t0, t1, integrator, steps=steps, downsample=self.conv4)          

		self.exps = None
		self.reset = False

		# To make predictions we use a fully connected layer (so convenient :))
		self.output = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
			Flatten(),
			nn.Linear(channels[3], classes)
		)

		# Define an optimizer so we can call it from the outside
		self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, nesterov=True) #, weight_decay=5e-4
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
		self.loss_module = nn.CrossEntropyLoss()

		# Check if GPU is available
		self.device = 'cpu'
		if torch.cuda.is_available():
			self.cuda()
			self.device = 'cuda'

		# Normalization
		self.normalize = False
		if isinstance(mu, list):
			self.mu = torch.tensor(mu).to(self.device)
			self.std = torch.tensor(std).to(self.device)
			self.normalize = True

		# Print the amount of parameters
		print("This model is using %d parameters" % (sum(p.numel() for p in self.parameters() if p.requires_grad)))
		print("This model is using %d Conv layers" % (len([m for m in self.modules() if isinstance(m, nn.Conv2d)])))

	def forward(self, x):
		# We do normalization in the model for the adversarial attacks to make sense.
		if self.normalize:
			x = (x - self.mu[None, :, None, None]) / self.std[None, :, None, None]

		x = self.solver1.integrate(x, reset=self.reset)
		x = self.solver2.integrate(x, reset=self.reset)
		x = self.solver3.integrate(x, reset=self.reset)
		x = self.solver4.integrate(x, reset=self.reset)

		if self.reset:
			self.reset = False

		if isinstance(self.solver1.exp, torch.Tensor):
			self.exps = sum([self.solver1.exp, self.solver2.exp, self.solver3.exp, self.solver4.exp])

		return self.output(x)