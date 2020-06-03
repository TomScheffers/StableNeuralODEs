# Overview of methods: https://en.wikipedia.org/wiki/List_of_Runge–Kutta_methods#Explicit_midpoint_method
import torch
import torch.nn.functional as F

# The forward Euler method is the most simple first order method
class Euler():
    name = "Euler"
    order = 1
    def step(f, t, dt, y):
        return y + dt * f(t, y)

# The Modified Euler uses an intermediate step
class ModifiedEuler():
    name = "ModifiedEuler"
    order = 2
    def step(f, t, dt, y):
        k1 = f(t, y)
        k2 = f(t + dt, y + dt * k1)
        return y + dt * (k1 + k2) / 2

# Runge Kutta 4 method
class RungeKutta4():
    name = "RungeKutta4"
    order = 4
    def step(f, t, dt, y):
        k1 = f(t, y)
        k2 = f(t + dt / 2,  y + dt * k1 / 2)
        k3 = f(t + dt / 2,  y + dt * k2 / 2)
        k4 = f(t + dt,      y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

class K2K3L2DistRungeKutta4():
    name = "K2K3L2DistRungeKutta4"
    order = 4
    def step(f, t, dt, y):
        k1 = f(t, y)
        k2 = f(t + dt / 2,  y + dt * k1 / 2)
        k3 = f(t + dt / 2,  y + dt * k2 / 2)
        k4 = f(t + dt,      y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, ((k2 - k3) ** 2).mean(0).sum().sqrt()

class K2K3L1DistRungeKutta4():
    name = "K2K3L1DistRungeKutta4"
    order = 4
    def step(f, t, dt, y):
        k1 = f(t, y)
        k2 = f(t + dt / 2,  y + dt * k1 / 2)
        k3 = f(t + dt / 2,  y + dt * k2 / 2)
        k4 = f(t + dt,      y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, torch.abs(k2 - k3).mean(0).sum()

class K2K3CosSimRungeKutta4():
    name = "K2K3CosSimRungeKutta4"
    order = 4
    def step(f, t, dt, y):
        k1 = f(t, y)
        k2 = f(t + dt / 2,  y + dt * k1 / 2)
        k3 = f(t + dt / 2,  y + dt * k2 / 2)
        k4 = f(t + dt,      y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6, F.cosine_similarity(k2.view(y.shape[0], -1), k3.view(y.shape[0], -1), dim=1, eps=1e-8).mean()