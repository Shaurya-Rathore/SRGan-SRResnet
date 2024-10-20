import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def get_ds():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((16,16)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader,testloader

import torch
from PIL import Image
from torch.utils.data import Dataset
import json
import os
import cv2

    

class StanfordDogsDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader for Stanford Dogs dataset.
    """

    def __init__(self, image_paths, transform=None, resize=(200, 200), crop_size=(200,200)):
        """
        :param image_paths: list of (image_path, label) tuples for the dataset split
        :param transform: optional transforms to be applied to the images
        """
        self.image_paths = image_paths
        self.transform = transform
        self.resize = resize
        self.crop_size = crop_size

        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        img = transforms.CenterCrop(self.crop_size)(img)
        
        if self.transform:
            img = self.transform(img)

        # Resize the image tensor
        lr_img = F.interpolate(img.unsqueeze(0), size=(100, 100), mode='bicubic', align_corners=False).squeeze(0)

        return lr_img, img

    def __len__(self):

        return len(self.image_paths)
    

class CIFAR10Dataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader for CIFAR-10 dataset.
    """

    def __init__(self, root, split, transform=None):
        """
        :param root: path to the CIFAR-10 dataset directory
        :param split: one of 'train' or 'test'
        :param transform: optional transforms to be applied to the images
        """

        self.root = root
        self.split = split.lower()
        self.transform = transform

        assert self.split in {'train', 'test'}

        if self.transform is None:
            self.transform = transforms.ToTensor()

        if self.split == 'train':
            self.data  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        else:
            self.data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    def __getitem__(self, i):
        img, target = self.data[i]

        # Resize the image tensor
        lr_img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(16, 16), mode='bicubic', align_corners=False).squeeze(0)

        return lr_img, img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.data)

import os
from sklearn.model_selection import train_test_split

def create_datasets(root_dir, split=0.2, train_val_split=0.2, transform=None):
    """
    Create train, validation, and test datasets from the Stanford Dogs dataset.

    :param root_dir: Root directory of the dataset.
    :param train_test_split: Ratio of test split.
    :param train_val_split: Ratio of validation split.
    :param transform: Optional transform to be applied to the images.
    :return: train_dataset, val_dataset, test_dataset
    """
    # Get list of all image file paths
    all_images = []
    for breed_name in os.listdir(root_dir):
        breed_dir = os.path.join(root_dir, breed_name)
        if os.path.isdir(breed_dir):
            for img_name in os.listdir(breed_dir):
                img_path = os.path.join(breed_dir, img_name)
                all_images.append((img_path, breed_name))

    print(len(all_images))

    # Split the dataset into train, val, and test
    train_val_images, test_images = train_test_split(all_images, test_size=split, random_state=42)
    train_images, val_images = train_test_split(train_val_images, test_size=train_val_split, random_state=42)

    # Define dataset objects for train, val, and test
    train_dataset = StanfordDogsDataset(train_images, transform=transform)
    val_dataset = StanfordDogsDataset(val_images, transform=transform)
    test_dataset = StanfordDogsDataset(test_images, transform=transform)

    return train_dataset, val_dataset, test_dataset

def denormalize(tensor, mean=(0.4761392,  0.45182742, 0.39101657), std=(0.23364353, 0.2289059,  0.22732813)):
    """Denormalize a tensor image with mean and standard deviation.
    
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    
    Returns:
        Tensor: Denormalized Tensor image.
    """
    denormalized_tensor = tensor.clone()
    # Iterate over channels and denormalize out-of-place
    for c in range(tensor.size(0)):
        denormalized_tensor[c] = tensor[c] * std[c] + mean[c]
    return denormalized_tensor