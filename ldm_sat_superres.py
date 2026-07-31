from setuptools import setup, find_packages
import os
import shutil
import nibabel as nib
from PIL import Image
import PIL
import SimpleITK as sitk
import gzip
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
import argparse, os, sys, glob,shutil
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.interp import *
from einops import rearrange, repeat
import cv2
from edsr import edsr 

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument(
        "--input_size",
        type=int,
        default=320,
        help="Size to which each image will be resized (square shape, e.g., 320x320)",
    )
    parser.add_argument(
        "-n",
        "--n_slides_generated",
        type=int,
        default=2,
        help="Number of slices generated between two consecutive slices (temporal resolution)",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta parameter for DDIM sampling",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="Number of steps for DDIM sampling",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config of the LDM model",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )

    return parser


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def get_number(element):
        return int(element.split('_')[0]) # THe order of the slices must be {Id of the order}_{Arbitrary Name}.png

if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(opt.input_size // 4),
        ])

    t_enc = int(opt.ddim_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = OmegaConf.load(f"{opt.config_path}")


################### LOADING LDM

    model = load_model_from_config(config, f"{opt.ckpt_path}")
    sampler = DDIMSampler(model)


#################### Taking inputs images (they should be in execution/inputs/) 


    images_z= sorted(os.listdir('execution/input/'),key=get_number)
    init_image_list = []
    for item in images_z:
                        cur_image = load_img(os.path.join('execution/input/', item)).to(device)
                        cur_image = transform(cur_image)
                        init_image_list.append(cur_image)
    init_image = torch.cat(init_image_list, dim=0)


#################### Interpolation in the latent space

  
shape = (3, opt.input_size // 4, opt.input_size // 4)
        
x_T = torch.randn(shape).to(device).unsqueeze(0)

for i in range(len(images_z)-1): 
           
            cond1 = init_image[i]
            cond2 = init_image[i+1]    
            
            sample_full, _ = sampler.sample(
                  t_enc,
                  2,
                  shape,
                  torch.cat([cond1.unsqueeze(0),cond2.unsqueeze(0)], dim=0),
                  eta=opt.ddim_eta,
                  x_T=torch.cat([x_T,x_T], dim=0),
                  unconditional_guidance_scale=1.0,
                  verbose=False,
                    temperature= 0.8
                    )
              

            for strategy in ["Slerp", "Lerp","LatentLerp","LatentSlerp"]:
            

              if strategy == "Slerp":
                
                for t in range(1,opt.n_slides_generated):
              
                  cond_interp = slerp_im(cond1, cond2,t/opt.n_slides_generated).unsqueeze(0)
                  
                  sample_interp, _ = sampler.sample(
                    t_enc,
                    1,
                    shape,
                    cond_interp,
                    eta=opt.ddim_eta,
                    x_T=x_T,
                    unconditional_guidance_scale=1.0,
                    verbose=False,
                    temperature= 0.8
                      )
              
                  x_interp = model.decode_first_stage(sample_interp[0].unsqueeze(0))
                  x_interp = torch.clamp((x_interp + 1.0) / 2.0, min=0.0, max=1.0)
                  im_interp = 255. * rearrange(x_interp[0].cpu().numpy(), 'c h w -> h w c')
                  final_np = np.clip(im_interp, 0, 255).astype(np.uint8)
                  Image.fromarray(final_np).save(os.path.join('execution/output/', f"DDIM_{images_z[i].split('.')[0]}_{t}_{strategy.lower()}.png"))
                    

              elif strategy == "Lerp":
                
                for t in range(1,opt.n_slides_generated):

                  cond_interp = ((1-t/opt.n_slides_generated)*cond1 + (t/opt.n_slides_generated)*cond2 ).unsqueeze(0)
                
                  sample_interp, _ = sampler.sample(
                  t_enc,
                  1,
                  shape,
                  cond_interp,
                  eta=opt.ddim_eta,
                  x_T=x_T,
                  unconditional_guidance_scale=1.0,
                  verbose=False,
                  temperature= 0.8
                    )
                    
                  x_samples_i = model.decode_first_stage(sample_interp[0].unsqueeze(0))
                  x_samples_i = torch.clamp((x_samples_i + 1.0) / 2.0, min=0.0, max=1.0)
                  x_samples_i = 255. * rearrange(x_samples_i[0].cpu().numpy(), 'c h w -> h w c')
                  final_np = np.clip(x_samples_i, 0, 255).astype(np.uint8)
                  Image.fromarray(final_np).save(os.path.join('execution/output/', f"DDIM_{images_z[i].split('.')[0]}_{t}_{strategy.lower()}.png"))
                
              
              elif strategy == "LatentLerp":
              
                for t in range(1,opt.n_slides_generated):
              
                  sample_interp = (1-t/opt.n_slides_generated)*sample_full[0]+(t/opt.n_slides_generated)*sample_full[1]
          
                  x_samples_i = model.decode_first_stage(sample_interp.unsqueeze(0))
                  x_samples_i = torch.clamp((x_samples_i + 1.0) / 2.0, min=0.0, max=1.0)
                  x_samples_i = 255. * rearrange(x_samples_i[0].cpu().numpy(), 'c h w -> h w c')
                  final_np = np.clip(x_samples_i, 0, 255).astype(np.uint8)
                  Image.fromarray(final_np).save(os.path.join('execution/output/', f"DDIM_{images_z[i].split('.')[0]}_{t}_{strategy.lower()}.png"))
                
              
              elif strategy == "LatentSlerp":
              
                for t in range(1,opt.n_slides_generated):

                  sample_interp = slerp(sample_full[0], sample_full[1],t/opt.n_slides_generated)
          
                  x_samples_i = model.decode_first_stage(sample_interp.unsqueeze(0))
                  x_samples_i = torch.clamp((x_samples_i + 1.0) / 2.0, min=0.0, max=1.0)
                  x_samples_i = 255. * rearrange(x_samples_i[0].cpu().numpy(), 'c h w -> h w c')
                  final_np = np.clip(x_samples_i, 0, 255).astype(np.uint8)
                  Image.fromarray(final_np).save(os.path.join('execution/output/', f"DDIM_{images_z[i].split('.')[0]}_{t}_{strategy.lower()}.png"))
                       
