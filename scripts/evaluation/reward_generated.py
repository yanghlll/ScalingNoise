import io
import os
import cv2
import json
import clip
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from omegaconf import OmegaConf

# CACHE_DIR = os.environ.get('VBENCH_CACHE_DIR')
# if CACHE_DIR is None:
#     CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'vbench')
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image


sub_model = None

def subject_consistency(video_list, device):
    global sub_model
    if sub_model == None:
        submodules_list = {'repo_or_dir': 'facebookresearch/dino:main', 'source': 'github', 'model': 'dino_vitb16', 'read_frame': None}
        
        dino_model = torch.hub.load(**submodules_list).to(device)
        sub_model = dino_model
    else:
        dino_model = sub_model
    sim = 0.0
    cnt = 0
    video_sim = 0
    images_list = [dino_transform_image_gpu(video_list[i].to(device), 224, device) for i in range(len(video_list))]

    with torch.no_grad():
        anchor_image = images_list[0].unsqueeze(0)
        anchor_image = anchor_image.to(device)
        anchor_features = dino_model(anchor_image)
        anchor_features = F.normalize(anchor_features, dim=-1, p=2)
   
    
    image_list = images_list[1:]
    for i in range(len(images_list)):
        with torch.no_grad():
            image = images_list[len(images_list)-1].unsqueeze(0)
            image = image.to(device)
            image_features = dino_model(image)
            image_features = F.normalize(image_features, dim=-1, p=2)
            if i == 0:
                sim_pre = max(0.0, F.cosine_similarity(anchor_features, image_features).item())
                cur_sim = sim_pre
                video_sim += cur_sim
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                sim_fir = max(0.0, F.cosine_similarity(anchor_features, image_features).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
        former_image_features = image_features
    sim_per_images = video_sim / (len(images_list) - 1)
    return sim_per_images
  

def dino_transform_image_gpu(batch_tensor, n_px, device):
    resized_tensor = resize(batch_tensor, (n_px, n_px), antialias=False)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device) 
    std = torch.tensor([0.229, 0.224, 0.225], device=device)   

    normalized_tensor = (resized_tensor - mean[:, None, None]) / std[:, None, None]

    return normalized_tensor