from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_mnist(batch_size=64):
    transform =  transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform), batch_size=batch_size,
        shuffle=True, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform), batch_size=batch_size*2, 
        shuffle=False, drop_last=True
    )
    return train_loader, test_loader

def get_cifar10(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    transform_test  = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=False, download=True, transform=transform_test), batch_size=batch_size*2, 
        shuffle=False, drop_last=True
    )
    return train_loader, test_loader

def get_cifar100(batch_size=64):
    #transform =  transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
    #])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
    ])

    transform_test  = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
    ])
    train_loader = DataLoader(
        datasets.CIFAR100(root='.data/cifar100', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR100(root='.data/cifar100', train=False, download=True, transform=transform_test), batch_size=batch_size*2, 
        shuffle=False, drop_last=True
    )
    return train_loader, test_loader

def get_imagenet(batch_size=64):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test  = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_loader = DataLoader(
        datasets.ImageNet(root='.data/imagenet', train=True, download=False, transform=transform_train), batch_size=batch_size,
        shuffle=True, drop_last=True
    )

    test_loader = DataLoader(
        datasets.ImageNet(root='.data/imagenet', train=False, download=False, transform=transform_test), batch_size=batch_size*2, 
        shuffle=False, drop_last=True
    )
    return train_loader, test_loader