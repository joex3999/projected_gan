import argparse
import math
import os

import torch
import numpy as np
import click 
import dnnlib
import legacy
import lpips

from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import List

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--imgfile',required=True)
def project(network_pkl: str, 
            imgfile:str ): 
    print("Starting to project image to latent space")

    ##Prepare Data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    resize= 256
    n_mean_latent = 10000
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []
    print("hey")
    print(imgfile)
    img = transform(Image.open(imgfile).convert("RGB"))
    imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    ## Generate Image
    if hasattr(G.synthesis, 'input'):
        m = make_transform('0,0', 0) ##TODO: Get from gen_images
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    with torch.no_grad():
        z = torch.randn(n_mean_latent, 512, device=device)
        latent_out = G.mapping(z,0) # ws

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
   
    percept = lpips.PerceptualLoss(
    model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    # z = torch.from_numpy(np.random.RandomState(1).randn(1, G.z_dim)).to(device).float()
    # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
if __name__ == "__main__" : 
    device= "cuda"
    project()