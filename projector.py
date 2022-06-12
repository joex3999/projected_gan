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
from typing import List,Tuple

def make_noise(log_size,device):

    noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

    for i in range(3, log_size + 1):
        for _ in range(2):
            noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

    return noises
def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


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

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp
#TODO: Add help arguments
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--imgfile',required=True)
@click.option('--lr',type=float,default=0.1)
@click.option('--step',type=int,default=1000)
@click.option('--noise',type=float,default=0.05)
@click.option('--noise_ramp',typle=float,default=0.75)
def project(network_pkl: str, 
            imgfile:str,
            lr:float,
            step:int,
            noise:float,
            noise_ramp:float): 
    print("Starting to project image to latent space")

    ##Prepare Data
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    resize= 256
    n_mean_latent = 10000
    truncation_psi=1 #What is truncation psi
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
        latent_out = G.mapping(z,0) # ws #Mapping and style from other implementation do not return same size ?

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.LPIPS(net="vgg")
    noises_signal=make_noise(int(np.log2(G.img_resolution)),device)
    noises=[]
    for noise in noises_signal:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
    print(imgs.shape[0])
    print(latent_mean.detach().clone().unsqueeze(0).shape)
    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1) ##TODO: Check Problem here 

    latent_in.requires_grad = True
   
    # if args.w_plus:
    #  latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1) #TODO: Enable w_plus latent inversion

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=lr)

    pbar = tqdm(range(step))
    latent_path = []
    label = torch.zeros([1, G.c_dim], device=device)
    for i in pbar:
        t = i / step
        lr = get_lr(t, lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * noise * max(0, 1 - t / noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        
        img_gen, _ = G([latent_n], input_is_latent=True, noise=noises)

        img = G([latent_n], label, truncation_psi=truncation_psi,noise=noises,input_is_latent=True)
        pbar.set_description((
            f"Hey Hey {i}"
        ))
    # z = torch.from_numpy(np.random.RandomState(1).randn(1, G.z_dim)).to(device).float()
    # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
if __name__ == "__main__" : 
    device= "cuda"
    project()