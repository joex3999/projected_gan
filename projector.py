import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import List
import click 
import dnnlib

import legacy
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('files',nargs="+", help='files to retireve latent code for', type=str, required=True, metavar='FILES')
def project(network_pkl: str, 
            files:List[str] ): 
    print("Starting to project image to latent space")

    ##Prepare Data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    resize= 256
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    for imgfile in files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    ## Generate Image
    if hasattr(G.synthesis, 'input'):
        m = make_transform('0,0', 0)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    z = torch.from_numpy(np.random.RandomState("01").randn(1, G.z_dim)).to(device).float()
    # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
if __name__ == "__main__" : 
    device= "cuda"
    project()