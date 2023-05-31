from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Glow tester")
parser.add_argument(
    "--n_flow", default=16, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=3, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--img_size", default=56, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=64, type=int, help="number of samples")


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def test(args, model):
    z_sample = []
    # z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)   # if img is RGB
    z_shapes = calc_z_shapes(1, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    with torch.no_grad():
        img_data = model.reverse(z_sample).cpu().data
        utils.save_image(
            img_data,
            f"mnist_sample.png",
            normalize=True,
            nrow=8,
            range=(-0.5, 0.5),
        )


def main():
    args = parser.parse_args()
    print(args)
    model = Glow(
        # 3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu  # if img is RGB
        1, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    # model.load_state_dict(torch.load('./checkpoint/best_model.pth'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./checkpoint/best_model.pth').items()})
    model = model.to(device)
    test(args, model)


if __name__ == '__main__':
    main()
