import os
import random
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import Model
import pandas as pd
import numpy as np

from config import CONFIG
import cv2
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainDataset(Dataset):

    def __init__(self, data_dir, df, seq_l, resize=None):

        self.resize = resize
        self.data_dir = data_dir
        
        self.df = df
        self.shifted_df = df[seq_l - 1:].reset_index()
        self.seq_l = seq_l

        self.transform = A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            A.GaussNoise(var_limit=1, p=0.5),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ToTensorV2()
        ])
                
    def __len__(self):
        return len(self.shifted_df)
     
    def __getitem__(self, index):
        
        start_index = self.shifted_df['index'].iloc[index]
        
        X = []
        all_steer = []
        
        
        for i in reversed(range(self.seq_l)):
            
            steer = self.df.steer.iloc[start_index - i]
            img_id = self.df.image_id.iloc[start_index - i]
            
            all_steer.append(steer)
            
            img_path = self.data_dir + img_id
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not self.resize is None:
                img = cv2.resize(img, self.resize)

            img = self.transform(image=img)["image"]

            X.append(img)
        
        X = torch.stack(X, dim=0)
         
        all_steer = np.clip(all_steer,-1.0, 1.0)
        
        steer_angles = torch.FloatTensor(all_steer[:-1])

        last_steer = torch.FloatTensor([all_steer[-1]])
        
        return X, last_steer, steer_angles

class TrainOversampledDataset(Dataset):

    def __init__(self, data_dir, df, os_df, seq_l):

        self.data_dir = data_dir
        
        self.df = df
        self.shifted_df = os_df[seq_l - 1:]
        self.seq_l = seq_l

        self.transform = A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),
            A.GaussNoise(var_limit=1, p=0.5),
            A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ToTensorV2()
        ])
                
    def __len__(self):
        return len(self.shifted_df)
     
    def __getitem__(self, index):
        
        start_index = self.shifted_df["index"].iloc[index]
        
        X = []
        all_steer = []
        
        for i in reversed(range(self.seq_l)):
            
            steer = self.df.steer.iloc[start_index - i]
            img_id = self.df.image_id.iloc[start_index - i]
            
            all_steer.append(steer)
            
            img_path = self.data_dir + img_id
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = self.transform(image=img)["image"]

            X.append(img)
        
        X = torch.stack(X, dim=0)
         
        all_steer = np.clip(all_steer,-1.0, 1.0)
        
        steer_angles = torch.FloatTensor(all_steer[:-1])

        last_steer = torch.FloatTensor([all_steer[-1]])
        
        return X, last_steer, steer_angles
    
class TestDataset(Dataset):

    def __init__(self, data_dir, df, resize=None):

        self.resize = resize
        self.data_dir = data_dir
        
        self.df = df.reset_index()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.df)
     
    def __getitem__(self, index):
        
        steer = self.df.steer.iloc[index]
        
        img_id = self.df.image_id.iloc[index]
        img_path = self.data_dir + img_id
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.resize is None:
            img = cv2.resize(img, self.resize)
        
        x = self.transform(img)
        
        return x, steer 
        
        
class DIPLECSTrainDataset(Dataset):

    def __init__(self, data_dir, prefix, df, seq_l):
    
        self.prefix = prefix
        self.data_dir = data_dir
        
        self.df = df
        self.shifted_df = df[seq_l - 1:].reset_index()
        self.seq_l = seq_l

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
    def __len__(self):
        return len(self.shifted_df)
     
    def __getitem__(self, index):
        
        start_index = self.shifted_df['index'].iloc[index]
        
        X = []
        all_steer = []

        for i in reversed(range(self.seq_l)):

            all_steer.append(self.df.data.loc[start_index - i])
            
            img_index = str(start_index - i + 1).zfill(6)

            img_path = self.data_dir + self.prefix + "Image" + img_index + '.jpg'
            img = cv2.imread(img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (320, 240,))
            
            x = self.transform(img)
            X.append(x)
        
        X = torch.stack(X, dim=0)
        
        all_steer = np.clip(all_steer,-1.0, 1.0)

        steer_angles = torch.FloatTensor(all_steer[:-1])
        last_steer = torch.FloatTensor([all_steer[-1]])
        
        return X, last_steer, steer_angles
    
    
class DIPLECSTestDataset(Dataset):

    def __init__(self, data_dir, prefix, df):

        self.prefix = prefix
        self.data_dir = data_dir
        
        self.df = df
        
        self.transform = transforms.Compose([
#             transforms.Resize(120,320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
         
    def __len__(self):
        return len(self.df)
     
    def __getitem__(self, index):
        

        img_index = self.df.iloc[index].image_id

        img_path = self.data_dir + self.prefix + img_index
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 240,))

        x = self.transform(img)
        
        steer = self.df.iloc[index].data
                
        return x, steer