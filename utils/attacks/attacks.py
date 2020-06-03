# From: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
import torch
import torch.nn.functional as F
from utils.training.training_ops import accuracy, Average

# Testing Gaussian noise attacks
def gaussian_noise_attack(model, test_loader, std):
    # Accuracy
    acc = Average()

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(model.device), target.to(model.device)

        # Add noise
        data += torch.empty_like(data).normal_(mean=0, std=std)

        # Forward pass the data through the model
        output = model(data)

        # Calculate accuracy
        acc.update(accuracy(output, target))

    # Calculate final accuracy for this epsilon
    print("White noise attack. Std: {}, Accuracy: {}".format(std, acc.eval()))

    # Return the accuracy and an adversarial example
    return acc.eval()

# Testing Salt and Pepper noise attacks:
def salt_and_pepper_attack(model, test_loader, rate):
    # Accuracy
    acc = Average()

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(model.device), target.to(model.device)

        # Add noise
        rand = torch.rand_like(data)
        data[(rand < rate/2)] = 0.
        data[(rand > 1 - rate/2)] = 1.

        # Forward pass the data through the model
        output = model(data)

        # Calculate accuracy
        acc.update(accuracy(output, target))

    # Calculate final accuracy for this epsilon
    print("Salt and pepper attack. Rate: {}, Accuracy: {}".format(rate, acc.eval()))

    # Return the accuracy and an adversarial example
    return acc.eval()

# Testing black box attack
# Inspired by: https://github.com/cg563/simple-blackbox-attack/blob/master/simba_single.py
def simba_attack(model, test_loader, batches=10, num_iters=200, epsilon=8/255.):
    # Accuracy
    acc = Average()

    # Get the dimensions
    data, target = next(iter(test_loader))
    n_dims = data[0].view(-1).size(0)
    perm = torch.randperm(n_dims)

    # Loop over all examples in test set
    with torch.no_grad():
        for data, target in test_loader:
            # Send the data and label to the device
            data, target = data.to(model.device), target.to(model.device)

            # Initial probability
            output = model(data)
            prob = F.softmax(output, 1).gather(1, target.view(-1, 1)).view(-1)

            for i in range(num_iters):
                # Make difference based on permutation
                diff = torch.zeros(n_dims).to(model.device)
                diff[perm[i]] = epsilon
                diff = diff.view(data[0].size())

                # Add / Subtract the difference
                data_add, data_sub = data + diff, data - diff

                # Forward pass the data through the model
                output_add = model(data_add)
                output_sub = model(data_sub)

                # Probabilities
                prob_add = F.softmax(output_add, 1).gather(1, target.view(-1, 1)).view(-1)
                prob_sub = F.softmax(output_sub, 1).gather(1, target.view(-1, 1)).view(-1)

                # Mask directions
                add = (prob_add < prob) * (prob_add < prob_sub)
                sub = (prob_sub < prob) * (prob_sub < prob_add)

                # Perturbate image
                data[add, ...] += diff
                data[sub, ...] -= diff

                # Clamp image
                data = data.clamp(0, 1)

                # Update probability
                prob[add] = prob_add[add]
                prob[sub] = prob_sub[sub]

            # Calculate accuracy
            batch_acc = accuracy(model(data), target)
            acc.update(batch_acc)

            if acc.N >= batches:
                # Calculate final accuracy for this epsilon
                print("SimBA attack: Iterations: {0}, Accuracy: {1}".format(num_iters, acc.eval()))

                # Return the accuracy and an adversarial example
                return acc.eval()