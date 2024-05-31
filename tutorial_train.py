import sys
sys.path.insert(0, '/home/ubuntu/miniconda3/envs/control/lib/python3.8/site-packages/')

import torch
assert torch.__version__ == '2.0.0+cu118', 'Wrong version of torch used' 

import xformers
assert xformers.__version__ == '0.0.18', 'Wrong version of xformers used' 

from share import *

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataframe_dataset import batch_to_pil, ColorDataModule
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import hashlib

from osTools import *
from strOps import *
from listOps import *

seed_everything(42)

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

# Configs
resume_log_dir = './lightning_logs/version_4' 
resume_path = './models/control_sd15_color_ini.ckpt' if resume_log_dir is None else resolve_checkpoint_from_default_pl_log_dir(resume_log_dir)
batch_size = 1
accumulate_grad_batches = 96
logger_freq = 8000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
every_n_train_steps = 1000

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_color.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))

print(f'Resuming from - {resume_path}, weight hash - {hash_model(model)}')

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
datamodule = ColorDataModule(batch_size=batch_size, num_workers=2)

logger = ImageLogger(batch_frequency=logger_freq, batch_to_pil=batch_to_pil, accumulate_grad_batches=accumulate_grad_batches // 8)
model_checkpointer = pl.callbacks.ModelCheckpoint(every_n_train_steps=every_n_train_steps, save_last=True)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, model_checkpointer], accumulate_grad_batches=accumulate_grad_batches)

# Train!
trainer.fit(model, datamodule, ckpt_path=resume_path)
