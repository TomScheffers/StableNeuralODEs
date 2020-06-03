import numpy as np

def accuracy(predictions, targets):
    targets = targets.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    return np.sum(1. * (np.argmax(predictions, axis=1) == targets)) / targets.shape[0]

def loader_accuracy(model, loader):
    correct, count = 0, 0
    for i, (x, y) in enumerate(loader):
        targets = y.cpu().detach().numpy()
        predictions = np.argmax(model(x.to(model.device)).cpu().detach().numpy(), axis=1)
        correct += np.sum(predictions == targets)
        count += y.shape[0]
    return correct / count

class Average(object):
    def __init__(self):
        self.reset()

    def update(self, value):
        self.N += 1
        self.T += value #1 / self.N * value + (1 - 1 / self.N) * self.mu

    def eval(self):
        return self.T / self.N if self.N > 0 else 0

    def reset(self):
        self.N = 0
        self.T = 0     
