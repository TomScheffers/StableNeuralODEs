# From: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
import torch
import torch.nn.functional as F
from utils.training.training_ops import accuracy, Average

# FGSM attack code
def fgsm(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# Testing FGSM attacks
def fgsm_attack(model, test_loader, epsilon):
    # Accuracy
    acc, snr = Average(), Average()

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(model.device), target.to(model.device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        if epsilon > 0:
            # Call FGSM Attack
            perturbed_data = fgsm(data, epsilon, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data)

        # Calculate accuracy
        acc.update(accuracy(output, target))
        for i in range(data.size(0)):
            snr.update(20 * torch.log10(data[i].norm() / (perturbed_data - data).norm()))
        print(snr.eval())

    # Calculate final accuracy for this epsilon
    print("Epsilon: {}, SNR: {}, Accuracy: {}".format(epsilon, snr.eval(), acc.eval()))

    # Return the accuracy and an adversarial example
    return acc.eval()