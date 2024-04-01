import argparse
from utils.SRResnet_Trainer import main

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

    main(config)
