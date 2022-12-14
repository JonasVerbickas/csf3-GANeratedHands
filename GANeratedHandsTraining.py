#!/usr/bin/env python
# coding: utf-8

import os
import sys
import wandb
import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from CycleGAN import CycleGAN
from HandDataModule import HandsDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import Helpers

# Check if GPU can be used
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == 'cpu':
    raise Exception("This shouldn't be ran on a CPU. Please, switch to a GPU/TPU env")

# Hyperparams for training
config = {
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 8,
    "NUM_EPOCHS": 300,
    "TRAIN_SPLIT": 0.8,
    "RAND_ROTATION": 0,
    "CPU_COUNT": int(os.cpu_count()),
    "TARGET_IMG_SIZE": (256, 256),
    "PIN_MEMORY": True if device == 'cuda' else False,
    "CYCLE_CONSISTENCY_WEIGHT": 10.0
}
# Save current config to wandb
wandb.init(project="CycleGAN", entity="jonasv", config=config)
wandb_logger = WandbLogger(log_model='all', project="CycleGAN")

# Open/setup dataset
dm = HandsDataModule(config)
dm.setup()
# Save a list of images that will ganerated and saved to wandb after each validation
# (this is a good way to observe how the training is going visually)
synth_for_visual_benchmarking, real_for_visual_benchmarking = Helpers.getImagesForVisualBenchmarking(dm, wandb_logger)
# Initialize model
model = CycleGAN(config, wandb_logger, synth_for_visual_benchmarking, real_for_visual_benchmarking)
print(ModelSummary(model))
# Initilize training
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=config['NUM_EPOCHS'],
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
