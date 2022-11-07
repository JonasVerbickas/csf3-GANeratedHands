#!/usr/bin/env python
# coding: utf-8

import os
import sys
import wandb
import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from OriginalCycleGAN.MyOriginalCycleGAN import CycleGAN as GAN
from HandDataModule import HandsDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# Check if GPU can be used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == 'cpu':
    raise Exception("This shouldn't be ran on a CPU. Please, switch to a GPU/TPU env")

# Save current config to wandb
config = {
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 8,
    "NUM_EPOCHS": 300,
    "TRAIN_SPLIT": 0.8,
    "CPU_COUNT": int(os.cpu_count()),
    "TARGET_IMG_SIZE": (256, 256),
    "PIN_MEMORY": True if device == 'cuda' else False,
    "CYCLE_CONSISTENCY_WEIGHT": 10.0
}
wandb.init(project="CycleGAN", entity="jonasv", config=config)
wandb_logger = WandbLogger(log_model='all', project="CycleGAN")

dm = HandsDataModule()
dm.setup()
list_of_images_for_visual_benchmarking = None
for i in dm.val_dataloader():
    list_of_images_for_visual_benchmarking = i[0]
    break
wandb_logger.log_image(key="original_images",
                       images=[T.ToPILImage()(img_tensor) for img_tensor in list_of_images_for_visual_benchmarking])

model = GAN(config, wandb_logger, list_of_images_for_visual_benchmarking)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=config['NUM_EPOCHS'],
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=wandb_logger,
    default_root_dir='./saved_models/'
)

# if an argument is given - assume that it's the path to model checkpoint
# load it and continue training it from the given checkpoint
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
    trainer.fit(model, dm, ckpt_path=checkpoint_path)
else:
    trainer.fit(model, dm)

path = "./saved_models/CycleGANeratedHandsv1.pth"
torch.save(model, path)
