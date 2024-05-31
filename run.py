import sys
sys.path.insert(0, '/home/ubuntu/miniconda3/envs/control/lib/python3.8/site-packages/')
import torch
import uuid

import xformers
assert xformers.__version__ == '0.0.18', 'Wrong version of xformers used' 

from share import *

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataframe_dataset import batch_to_pil, ColorDataModule
import torchvision
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import hashlib

from osTools import *
from strOps import *
from listOps import *

import gradio as gr 
from imageOps import *
from dataframe_dataset import *

def hash_model(model):
    model.eval()
    hasher = hashlib.sha256()
    state_dict = model.state_dict()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode('utf-8'))
        hasher.update(state_dict[key].cpu().numpy().tobytes())
    return hasher.hexdigest()

def resolve_checkpoint_from_default_pl_log_dir (log_dir) : 
    paths = list(allFilesWithSuffix(log_dir, 'ckpt'))
    last_ckpt_maybe = next(filter(lambda x : 'last' in x, paths), None)
    if last_ckpt_maybe is not None :
        return last_ckpt_maybe
    steps = [reverse_f_string(osp.split(_)[1], 'epoch={epoch}-step={step}', int)['step'] for _ in paths] 
    path = paths[argmax(steps)]
    print('Using checkpoint path', path)
    return path

resume_log_dir = './lightning_logs/version_4' 
resume_path = './models/control_sd15_color_ini.ckpt' if resume_log_dir is None else resolve_checkpoint_from_default_pl_log_dir(resume_log_dir)
model = create_model('./models/cldm_color.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.cuda()
model.eval()

def process (image_editor, prompt) :
    uniq_id = str(uuid.uuid4())

    img = imgArrayToPIL(image_editor['background']).convert('RGB')
    img = aspectRatioPreservingResizePIL(img, 512) 
    img = transforms.CenterCrop(512)(img) 
    img.save(osp.join('logs', f'{uniq_id}_image.png'))

    target = np.array(img)
    target = (target.astype(np.float32) / 127.5) - 1.0

    layers = [imgArrayToPIL(_) for _ in image_editor['layers']]
    ch_img = layers[0] 
    for im in layers[1:]: 
        ch_img = Image.alpha_composite(ch_img, im)

    ch_img = aspectRatioPreservingResizePIL(ch_img, 512) 
    ch_img = transforms.CenterCrop(512)(ch_img) 
    ch_img.save(osp.join('logs', f'{uniq_id}_color_hint.png'))

    edge_map = extract_edge_map(img, target_size=512) # PIL
    edge_map_np = np.array(edge_map) # [H, W] 

    color_hint_np = np.array(ch_img) # [H, W, 4]

    H, W = edge_map_np.shape

    # combine the two 
    hint = np.concatenate((edge_map_np.reshape(H, W, 1), color_hint_np), axis=2) # [H, W, 5], np.uint8 hopefully!
    hint = hint.astype(np.float32) / 255.0

    target = torch.from_numpy(target).unsqueeze(0).cuda().repeat(4, 1, 1, 1)
    hint = torch.from_numpy(hint).unsqueeze(0).cuda().repeat(4, 1, 1, 1)

    batch = {'jpg': target, 'txt': [prompt] * 4, 'hint': hint}

    images = model.log_images(batch, ddim_steps=25)

    k = 'samples_cfg_scale_9.00'
    N = images[k].shape[0]
    images[k] = images[k][:N]
    if isinstance(images[k], torch.Tensor):
        images[k] = images[k].detach().cpu()
        if True:
            images[k] = torch.clamp(images[k], -1., 1.)

    grid = torchvision.utils.make_grid(images[k], nrow=2)
    if True :
        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    result = Image.fromarray(grid)
    result.save(osp.join('logs', f'{uniq_id}_out.png'))

    return [result]

with gr.Blocks() as demo:
    with gr.Row() : 
        gr.Markdown("## ControlNet for Coloring Images") 

    with gr.Row() : 
        with gr.Column() : 
            image_editor = gr.ImageEditor()
            prompt = gr.Textbox(label="Prompt") 
            run_button = gr.Button()

        with gr.Column() : 
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery")

    run_button.click(fn=process, inputs=[image_editor, prompt], outputs=[result_gallery])

demo.launch(server_name='0.0.0.0', share=True)
