# From: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.training.training_ops import accuracy, Average

def pgd(model, data, target, eps, alpha, iters):
    images = data.to(model.device)
    labels = target.to(model.device)
    loss = nn.CrossEntropyLoss()
    ori_images = images.clone().detach()
    
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)
        cost = loss(outputs, labels).to(model.device)
        
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    adv_images = images
    return adv_images

# Testing PGD attacks
def pgd_attack(model, test_loader, epsilon=4/255., alpha=1/255., iters=20):
    # Accuracy
    acc = Average()

    # Loop over all examples in test set
    for data, target in test_loader:
        # Adv. images 
        perturbed_data = pgd(model, data, target, epsilon, alpha, iters)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Calculate accuracy
        acc.update(accuracy(output, target))

    # Calculate final accuracy for this epsilon
    print("PGD Attack. Epsilon: {0}, Alpha: {1}, Iterations: {2}, Accuracy: {3}".format(epsilon, alpha, iters, acc.eval()))

    # Return the accuracy and an adversarial example
    return acc.eval()