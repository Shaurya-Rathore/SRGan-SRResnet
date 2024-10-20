import argparse
from utils.SRGan_Trainer import validate
from utils.model import Generator
from utils.dataset import create_datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


def get_config(args):
    config= {
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling_factor', type=int,required=False, default=2)
    parser.add_argument('--large_kernel_size_g', type=int,required=False, default=9)
    parser.add_argument('--small_kernel_size_g', type=int,required=False, default=3)
    parser.add_argument('--n_blocks_g', type=int,required=False, default=6)
    parser.add_argument('--n_channels_g', type=int,required=False, default=32)
    parser.add_argument('--n_blocks_d', type=int,required=False, default=8)
    parser.add_argument('--kernel_size_d', type=int,required=False, default=3)
    parser.add_argument('--fc_size_d', type=int,required=False, default=512)
    parser.add_argument('--n_channels_d', type=int,required=False, default=32)
    parser.add_argument('--batch_size', type=int,required=False, default=16)
    parser.add_argument('--iterations', type=int,required=False, default=2e5)
    parser.add_argument('--beta', type=int,required=False, default=1e-3)
    parser.add_argument('--lr', type=float,required=False, default=1e-4)
    arguments = parser.parse_args()
    config = get_config(arguments)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('checkpoint_srgan.pth.tar')

    
    generator = Generator(large_kernel_size=config['large_kernel_size_g'],
                              small_kernel_size=config['small_kernel_size_g'],
                              n_channels=config['n_channels_g'],
                              n_blocks=config['n_blocks_g'],
                              scaling_factor=config['scaling_factor'])
    
    generator.load_state_dict(checkpoint['generator'])

    
    train_dataset, val_dataset, test_dataset = create_datasets(root_dir="/kaggle/input/stanford-dogs-dataset/images/Images",
                                                                split=0.2,
                                                            train_val_split=0.2,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),  # Convert PIL image to Tensor
                                                                transforms.Lambda(lambda img: img / 255.0),
  # Normalize pixel values to [-1, 1]
                                                            ]))
    
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle= True)
    
    validate(generator, test_loader, device)