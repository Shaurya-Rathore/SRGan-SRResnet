import torch
import argparse
import cv2
import streamlit as st
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.SRGan_Trainer import create_datasets, denormalize
from utils.model import Generator
import numpy as np
def validate(generator, dataloader, device, save_dir):
    generator.to(device)
    generator.eval()
    
    with torch.no_grad():
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            sr_imgs = generator(lr_imgs)

            # Denormalize images for proper visualization
            hr_img_denorm = denormalize(hr_imgs[0])
            sr_img_denorm = denormalize(sr_imgs[0])
            lr_img_denorm = denormalize(lr_imgs[0])

            # Convert tensors to PIL images
            hr_img_pil = TF.to_pil_image(hr_img_denorm.cpu())
            sr_img_pil = TF.to_pil_image(sr_img_denorm.cpu())
            lr_img_pil = TF.to_pil_image(lr_img_denorm.cpu())

            # Save images using cv2
            hr_img_path = f'{save_dir}/hr_image_{i}.png'
            sr_img_path = f'{save_dir}/sr_image_{i}.png'
            lr_img_path = f'{save_dir}/lr_image_{i}.png'

            hr_img_np = cv2.cvtColor(np.array(hr_img_pil), cv2.COLOR_RGB2BGR)
            sr_img_np = cv2.cvtColor(np.array(sr_img_pil), cv2.COLOR_RGB2BGR)
            lr_img_np = cv2.cvtColor(np.array(lr_img_pil), cv2.COLOR_RGB2BGR)

            cv2.imwrite(hr_img_path, hr_img_np)
            cv2.imwrite(sr_img_path, sr_img_np)
            cv2.imwrite(lr_img_path, lr_img_np)

            # Display images using Streamlit
            st.image([lr_img_pil, sr_img_pil, hr_img_pil], caption=["Low-Res", "Super-Res", "High-Res"], width=200)
            
            # You can add a delay for demonstration purposes
            st.write(f"Displaying results for image {i + 1}")

def get_config(args):
    config = {
        'scaling_factor': args.scaling_factor,
        'large_kernel_size_g': args.large_kernel_size_g,
        'small_kernel_size_g': args.small_kernel_size_g,
        'n_blocks_g': args.n_blocks_g,
        'n_channels_g': args.n_channels_g,
        'n_blocks_d': args.n_blocks_d,
        'n_channels_d': args.n_channels_d,
        'fc_size_d': args.fc_size_d,
        'kernel_size_d': args.kernel_size_d,
        'batch_size': args.batch_size,
        'beta': args.beta,
        'lr': args.lr,
        'iterations': args.iterations
    }
    return config

def predict(root_dir, save_dir='output_images'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling_factor', type=int, default=2)
    parser.add_argument('--large_kernel_size_g', type=int, default=9)
    parser.add_argument('--small_kernel_size_g', type=int, default=3)
    parser.add_argument('--n_blocks_g', type=int, default=6)
    parser.add_argument('--n_channels_g', type=int, default=32)
    parser.add_argument('--n_blocks_d', type=int, default=8)
    parser.add_argument('--kernel_size_d', type=int, default=3)
    parser.add_argument('--fc_size_d', type=int, default=512)
    parser.add_argument('--n_channels_d', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=int(2e5))
    parser.add_argument('--beta', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args([])
    config = get_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('checkpoint_srgan.pth.tar')

    generator = Generator(
        large_kernel_size=config['large_kernel_size_g'],
        small_kernel_size=config['small_kernel_size_g'],
        n_channels=config['n_channels_g'],
        n_blocks=config['n_blocks_g'],
        scaling_factor=config['scaling_factor']
    )
    generator.load_state_dict(checkpoint['generator'])

    train_dataset, val_dataset, test_dataset = create_datasets(
        root_dir=root_dir,
        split=0.2,
        train_val_split=0.2,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img / 255.0),
        ])
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    validate(generator, test_loader, device, save_dir)

def main():
    st.title('Super Resolution Show')
    file_dir = st.text_input('Dataset File Path')
    if st.button('Begin Show'):
        if file_dir:
            st.write("Processing the dataset...")
            predict(file_dir)
        else:
            st.error("Please provide a valid dataset path.")

if __name__ == '__main__':
    main()
