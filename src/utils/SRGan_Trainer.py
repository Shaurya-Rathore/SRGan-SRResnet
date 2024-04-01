import time
import torch.backends.cudnn as cudnn
from torch import nn
from utils.model import Generator, Discriminator
from utils.dataset import StanfordDogsDataset, create_datasets,denormalize
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import argparse
import cv2
import wandb

wandb.init(project='SRGan', entity='shaurya24')

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
start_epoch = 0  # start at this epoch
grad_clip = None 

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True

def validate(generator, dataloader, device):

    generator.to(device)
    generator.eval()
    
    with torch.no_grad():
       for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = generator(lr_imgs)
            
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

        # Generator
    generator = Generator(large_kernel_size=config['large_kernel_size_g'],
                              small_kernel_size=config['small_kernel_size_g'],
                              n_channels=config['n_channels_g'],
                              n_blocks=config['n_blocks_g'],
                              scaling_factor=config['scaling_factor'])

        # Initialize generator's optimizer
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=config['lr'])

        # Discriminator
    discriminator = Discriminator(kernel_size=config['kernel_size_d'],
                                      n_channels=config['n_channels_d'],
                                      n_blocks=config['n_blocks_d'],
                                      fc_size=config['fc_size_d'])

        # Initialize discriminator's optimizer
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=config['lr'])


    # Loss functions
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Custom dataloaders
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

        best_val_psnr = 0

        # One epoch's training
        train(train_loader=train_loader,
                                                generator=generator,
                                                discriminator=discriminator,
                                                content_loss_criterion=content_loss_criterion,
                                                adversarial_loss_criterion=adversarial_loss_criterion,
                                                optimizer_g=optimizer_g,
                                                optimizer_d=optimizer_d,
                                                epoch=epoch, config = config)
        

        
        new_psnr = validate(generator, val_loader, device)
        if (new_psnr >= best_val_psnr):
            # Save checkpoint
            best_val_psnr = new_psnr
            torch.save({'epoch': epoch,
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict()},
                    'checkpoint_srgan.pth.tar')


def train(train_loader, generator, discriminator, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch, config):
    """
    One epoch's training.

    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    print_freq = 50 
    losses_c = []  # content loss
    losses_a = []  # adversarial loss in the generator
    losses_d = []  # adversarial loss in the discriminator

    # Batches
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch') as pbar:
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            # Move to default device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # GENERATOR UPDATE

            # Generate
            sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

            # Discriminate super-resolved (SR) images
            sr_discriminated = discriminator(sr_imgs)  # (N)

            # Calculate the Perceptual loss
            content_loss = content_loss_criterion(sr_imgs, hr_imgs)
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
            perceptual_loss = content_loss + config['beta'] * adversarial_loss

            # Back-prop.
            optimizer_g.zero_grad()
            perceptual_loss.backward()


            # Update generator
            optimizer_g.step()

            # Keep track of loss
            losses_c.append(content_loss.item())
            losses_a.append(adversarial_loss.item())

            # DISCRIMINATOR UPDATE

            # Discriminate super-resolution (SR) and high-resolution (HR) images
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())
            # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
            # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
            # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
            # See FAQ section in the tutorial

            # Binary Cross-Entropy loss
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                            adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

            # Back-prop.
            optimizer_d.zero_grad()
            adversarial_loss.backward()

            # Update discriminator
            optimizer_d.step()

            # Keep track of loss
            losses_d.append(adversarial_loss.item())

            # Make the bar move
            pbar.set_postfix({'loss adv': losses_a[-1],'loss content': losses_c[-1]})
            pbar.update()
            wandb.log({"Content Loss": losses_c[-1], "Generator Adversarial Loss": losses_a[-1], "Discriminator Adversarial Loss": losses_d[-1]})


            # Print status
            

        del lr_imgs, hr_imgs, sr_imgs, hr_discriminated, sr_discriminated  # free some memory since their histories may be stored
