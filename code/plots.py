import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Make the training / testing loaders
from utils.datasets import get_cifar10
_,test_loader = get_cifar10(batch_size=128)
mu, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

# Define the model
from solvers.fixed_step import FixedStepSolver, FixedStepNumericalLyapunovSolverV2
from integrators.simple import Euler, ModifiedEuler, RungeKutta4
from networks.odenet_2 import OdeNet

import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF

checkpoints = [
    ("FixedStepSolver-Euler-2Step-Run0.pth", Euler, 2),
    ("FixedStepNumericalLyapunovSolverV2-Euler-2Step-leReg0.1-Vecs2-Run0.pth", Euler, 2),
    ("FixedStepRK4RegSolver-K2K3L2DistRungeKutta4-2Step-leReg0.01-Vecs0-Run0.pth", RungeKutta4, 1),
]

idxs = [0, 160, 40, 190, 215] #[random.randint(0, 256) for _ in range(5)] #
size = 128

width, height = (1 + len(checkpoints)) * size, size * len(idxs) 
grid = Image.new('RGB', (width, height))

with torch.no_grad():
    data, target = next(iter(test_loader))

    for (path, intr, steps) in checkpoints:
        model = OdeNet(solver=FixedStepNumericalLyapunovSolverV2, integrator=intr, t0=0.0, t1=1.0, steps=steps, in_channels=3, channels=[64, 128, 256, 512], classes=10,  mu=mu, std=std)

        print("\nCheckpoint: ", path)
        model.load_state_dict(torch.load("checkpoints/cifar10/"+path, map_location=torch.device('cpu'))["state_dict"])

        # Set loops higher to converge
        model.solver1.loops, model.solver2.loops, model.solver3.loops, model.solver4.loops = 10, 10, 10, 10

        # Convert data proper device, forward pass and calculate loss
        data, target = data.to(model.device), target.to(model.device)
        pred = model(data)
        for i in idxs:
            image = data[i].to('cpu').clone()
            img = TF.to_pil_image(image).resize((size, size), Image.ANTIALIAS)
            grid.paste(img, (0, idxs.index(i) * size))

            # Get vectors
            diff = model.solver2.diff[0][i, :, :, :].abs().mean(0).to('cpu')
            print(i, diff.sum(), diff.max())

            # Scale diff 0 - 1
            diff = diff / 1.3

            # Make heatmap
            heatmap = torch.zeros(size=[3] + list(diff.size()))
            heatmap[0, :, :] = 2 * diff
            heatmap[1, :, :] = 1 - 2 * torch.abs(diff - 0.5)
            heatmap[2, :, :] = 1 - 2 * diff
            heatmap = heatmap.clamp(0, 1)

            h_img = TF.to_pil_image(heatmap).resize((size, size), Image.ANTIALIAS)
            res = Image.blend(img, h_img, alpha=0.6)
            grid.paste(res, (size + checkpoints.index((path, intr, steps)) * size, idxs.index(i) * size))

    grid.show()
    grid.save('figs/grid_test.jpg')


