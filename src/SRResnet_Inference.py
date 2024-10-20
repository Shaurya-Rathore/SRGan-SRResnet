import argparse
from utils.SRResnet_Trainer import validate
from utils.model import SRResNet
import torch
from torch.utils.data import DataLoader
from utils.dataset import create_datasets
import torchvision.transforms as transforms


def get_config(args):
    config= {
        'scaling_factor': args.scaling_factor,
        'large_kernel_size': args.large_kernel_size,
        'small_kernel_size': args.small_kernel_size,
        'n_blocks': args.n_blocks,
        'n_channels': args.n_channels,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'iterations': args.iterations
    }
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling_factor', type=int,required=False, default=2)
    parser.add_argument('--large_kernel_size', type=int,required=False, default=9)
    parser.add_argument('--small_kernel_size', type=int,required=False, default=3)
    parser.add_argument('--n_blocks', type=int,required=False, default=6)
    parser.add_argument('--n_channels', type=int,required=False, default=32)
    parser.add_argument('--batch_size', type=int,required=False, default=16)
    parser.add_argument('--iterations', type=int,required=False, default=1e6)
    parser.add_argument('--lr', type=float,required=False, default=1e-4)
    arguments = parser.parse_args()
    config = get_config(arguments)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('checkpoint_srresnet.pth.tar')


    model = SRResNet(large_kernel_size=config['large_kernel_size'], small_kernel_size=config['small_kernel_size'],
                         n_channels=config['n_channels'], n_blocks=config["n_blocks"], scaling_factor=config['scaling_factor'])
    
    model.load_state_dict(checkpoint['model'])
    
    train_dataset, val_dataset, test_dataset = create_datasets(root_dir="/kaggle/input/stanford-dogs-dataset/images/Images",
                                                                split=0.2,
                                                            train_val_split=0.2,
                                                            transform=transforms.Compose([
                                                                transforms.ToTensor(),  # Convert PIL image to Tensor
                                                                transforms.Lambda(lambda img: img / 255.0),
                                                                transforms.Normalize((0.4761392,  0.45182742, 0.39101657),
                                                                                     (0.23364353, 0.2289059,  0.22732813))  # Normalize pixel values to [-1, 1]
                                                            ]))
    
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle= True)

    validate(model, test_loader, device)
