from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
# from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm
import os
os.environ['TORCH_HOME']='/data/kongweili/plug-and-play-main/pretrained_pth'

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

def process(input_image_path, output_image_path, prompt, a_prompt, n_prompt, num_samples, 
            image_resolution, ddim_steps, guess_mode, strength, scale, seed, low_threshold, high_threshold, eta):
    with torch.no_grad():
        input_image = cv2.imread(input_image_path)
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        H, W, C = img.shape

        # # Save detected_map
        # cv2.imwrite(detected_map_path, detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Save the output images
        for i in range(num_samples):
            cv2.imwrite(output_image_path.format(i), x_samples[i])


# 定义每个idx对应的folder数量
num_folders = {0: 8, 1: 12}
for i in range(2, 21):
    num_folders[i] = 10

# 定义每个idx对应的prompt
prompts = {}
for i in range(0, 2):
    prompts[i] = 'a photo of a scene in spring, best quality, extremely detailed'
for i in range(2, 12):
    prompts[i] = 'a photo of a scene in winter, best quality, extremely detailed'
for i in range(12, 21):
    prompts[i] = 'a photo of a scene in autumn, best quality, extremely detailed'


# Specify the paths and parameters
# input_image_path = '/data/horse.png'
# input_image_path = 'experiments/scene_in_spring/samples/0.png'
# detected_map_path = 'detected_map.png'
# output_image_path = 'output_image_{}.png'  # {} will be replaced by the index of the image
# prompt = 'a photo of a scene in winter'
a_prompt = ''
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 1
image_resolution = 512
# detect_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1.0
scale = 7.5
seed = -1
eta = 0.5
low_threshold = 100
high_threshold = 200


# 遍历每个idx
for idx in tqdm(range(21), desc="Processing batches", leave=False):
    # 遍历当前idx下的每个folder
    for folder in tqdm(range(1, num_folders[idx] + 1), desc=f"Processing images in batch {idx}", leave=False):
        # 定义路径和参数
        input_image_path = f'../StyleID-main/tests/batch_{idx}/{folder}/samples/0.jpg'
        output_image_path = f'../StyleID-main/tests/batch_{idx}/{folder}/cn_translations/{folder}_translated.jpg'  # {} will be replaced by the index of the image
        prompt = prompts[idx]

        # 调用process函数
        process(input_image_path, output_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, low_threshold, high_threshold, eta)


# # Call the process function
# process(input_image_path, detected_map_path, output_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, low_threshold, high_threshold, eta)