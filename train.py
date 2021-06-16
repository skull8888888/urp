import os
import random
import torch
import torchvision
import yaml
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import Trainer, seed_everything
from model import Model
import pandas as pd
import numpy as np
import math
from pytorch_lightning.callbacks import ModelCheckpoint
from config import CONFIG
from dataset import TrainDataset, TrainOversampledDataset, TestDataset, DIPLECSTrainDataset, DIPLECSTestDataset
from attrdict import AttrDict

seed_everything(42)

def main():
    
    with open("./hparams.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)  # config is dict
        cfg = AttrDict(config)
    
    batch_size = cfg.batch_size
    seq_l = cfg.seq_l
    epochs = cfg.epochs

    num_workers = 11
    
#     # f1t dataset
#     train_df = pd.read_csv('data/train_norm_denoised.csv')
#     os_df = pd.read_csv('data/train_os.csv')
#     val_df = pd.read_csv('data/val_norm_denoised.csv')

#     train_dataset = TrainOversampledDataset('data/train/', train_df, os_df, seq_l)
#     val_dataset = TestDataset('data/val/', val_df)
    
     # f1t dataset
    train_df = pd.read_csv('./PShape/train_1315584123/1315584123.csv')
    val_df = pd.read_csv('./PShape/val_1315584542/1315584542.csv')

    train_dataset = TrainDataset('./PShape/train_1315584123/1315584123', train_df, seq_l, resize=(320, 240))
    val_dataset = TestDataset('./PShape/val_1315584542/1315584542', val_df, resize=(320, 240))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers, 
        pin_memory=True)


    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers, 
        pin_memory=True)

    model = Model(cfg, len(train_loader))
#     model = Model.load_from_checkpoint(checkpoint_path="lightning_logs/version_21/checkpoints/epoch=18-val_loss=0.0836.ckpt", cfg=cfg, train_loader_len=len(train_loader), strict=False)
    model.train()

    val_acc_callback = ModelCheckpoint(
        monitor='val_loss', 
        filename='{epoch:02d}-{val_loss:.4f}',
        save_last=True, 
        mode='min',
        save_weights_only=True)

    
    trainer = Trainer(
        gpus=[0], 
        accelerator='ddp',
        callbacks=[val_acc_callback], 
        max_epochs=epochs,
        precision=16,
        num_sanity_val_steps=seq_l * 2)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()