#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 17:29:21
@Desc    :   Cleaned up batch inference template
@Ref     :   
'''
import time
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from controlnet_aux.processor import Processor
from diffusers import DDIMScheduler, AutoencoderKL
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from models.pipeline_controlvideo import ControlVideoPipeline
from models.util import save_videos_grid, read_video
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet


POS_PROMPT = (
    " ,best quality, extremely detailed, HD, ultra, 8K, HQ, masterpiece, trending on artstation, art, smooth")
NEG_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, "
    "low quality, deformed body, bloated, ugly, blurry, low res, unaesthetic"
)

data_root = '/data/trc/videdit-benchmark/DynEdit'
method_name = 'controlvideo'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config.yaml',
    output_dir=f'{data_root}/outputs/{method_name}',
    seed=33,
    # TODO define arguments
    sd_path='/data/trc/tmp-swh/models/stable-diffusion-v1-5',
    cn_path="/data/trc/tmp-swh/models/control_v11f1p_sd15_depth",
    inter_path = "checkpoints/flownet.pkl",
    smoother_steps=[19, 20],
))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # load model
    print('Loading models ...')
    device = torch.device('cuda')
    # TODO define model
    t2i_transform = torchvision.transforms.ToPILImage()
    processor = Processor('depth_midas')
    tokenizer = CLIPTokenizer.from_pretrained(config.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(config.sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(config.sd_path, subfolder="unet").to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(config.cn_path).to(dtype=torch.float16)
    interpolater = IFNet(ckpt_path=config.inter_path).to(dtype=torch.float16)
    scheduler=DDIMScheduler.from_pretrained(config.sd_path, subfolder="scheduler")
    pipe = ControlVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet, interpolater=interpolater, scheduler=scheduler,
    )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    data_config = OmegaConf.load(config.config_file)
    set_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/videos/{row.video_id}.mp4'
        # TODO load video
        video = read_video(video_path=video_path, video_length=24, width=512, height=512, frame_rate=None)

        # # Optional
        # inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        # inverse_path.mkdir(parents=True, exist_ok=True)
        
        # preprocess
        start = time.perf_counter()
        # TODO preprocess video
        pil_annotation = []
        for frame in video:
            pil_frame = t2i_transform(frame)
            pil_annotation.append(processor(pil_frame, to_pil=True))
        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)

        # edit
        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # TODO edit
            # prompts=edit['prompt'],
            # negative_prompts=edit['src_words']+negative_prompt,
            # inversion_prompt=row['prompt'],
            # edit['tgt_words']
            sample = pipe(
                edit['prompt'] + POS_PROMPT, video_length=24, frames=pil_annotation, 
                num_inference_steps=50, smooth_steps=config.smoother_steps,
                generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                width=512, height=512
            ).videos
            save_videos_grid(sample, output_dir / f"{i}.mp4", fps=12)
        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()