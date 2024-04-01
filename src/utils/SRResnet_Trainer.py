import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from utils.model import SRResNet
from utils.dataset import get_ds, StanfordDogsDataset, create_datasets, denormalize
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import wandb

wandb.init(project='SRResNet', entity='shaurya24')

# Model parameters

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
start_epoch = 0  # start at this epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

cudnn.benchmark = True

def validate(model, dataloader, device):

    model.to(device)
    model.eval()
    
    with torch.no_grad():
       for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            
            sr_img_psnr = sr_imgs.cpu().detach().numpy()
            hr_img_psnr = hr_imgs.cpu().detach().numpy()

            hr_img_denorm = denormalize(hr_imgs[0])
            sr_img_denorm = denormalize(sr_imgs[0])
            lr_img_denorm = denormalize(lr_imgs[0])

            hr_img_np = TF.to_pil_image(hr_img_denorm.cpu())
            sr_img_np = TF.to_pil_image(sr_img_denorm.cpu())
            lr_img_np = TF.to_pil_image(lr_img_denorm.cpu())   
                   
            sr_img_psnr = sr_imgs[0].cpu().detach().numpy()
            hr_img_psnr = hr_imgs[0].cpu().detach().numpy()
            print(cv2.PSNR(hr_img_psnr, sr_img_psnr))
                    # Plot images

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(hr_img_np)
            plt.title('HR Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(lr_img_np)
            plt.title('LR Image')
            plt.axis('off')
                    
            plt.subplot(1, 3, 3)
            plt.imshow(sr_img_np)
            plt.title('SR Image')
            plt.axis('off')

            plt.show()
        
            
            return cv2.PSNR(hr_img_psnr, sr_img_psnr)

def main(config):
    """
    Training.
    """
    global start_epoch, epoch, checkpoint
    

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SRResNet(large_kernel_size=config['large_kernel_size'], small_kernel_size=config['small_kernel_size'],
                         n_channels=config['n_channels'], n_blocks=config["n_blocks"], scaling_factor=config['scaling_factor'])
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config['lr'])

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    train_dataset, val_dataset, test_dataset = create_datasets(root_dir="/kaggle/input/stanford-dogs-dataset/images/Images",
                                                                split=0.2,
                                                            train_val_split=0.2,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),  # Convert PIL image to Tensor
                                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                                     (0.5, 0.5, 0.5))  # Normalize pixel values to [-1, 1]
                                                            ]))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    # Total number of epochs to train for
    epochs = int(config['iterations'] // len(train_loader) + 1)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        best_psnr = 0
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        
        new_psnr = validate(model, val_loader, device)
        
        if(new_psnr >= best_psnr):
        # Save checkpoint
            best_psnr = new_psnr
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       'checkpoint_srresnet.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization
    print_freq = 100 
    start = time.time()
    losses = []


    # Batches
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch') as pbar:
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            # Move to default device
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]
            # Forward prop.
            sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

            # Loss
            loss = criterion(sr_imgs, hr_imgs)  # scalar

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Update model
            optimizer.step()

            # Keep track of loss
            losses.append(loss.item())

            # Update tqdm progress bar
            pbar.set_postfix({'loss': loss.item()})
            pbar.update()

            del lr_imgs, hr_imgs, sr_imgs
            wandb.log({"Loss": loss.item()})
