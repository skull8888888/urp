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

class TrainDataset(Dataset):

    def __init__(self, data_dir, df, seq_l, train=True):

        self.data_dir = data_dir
        
        self.df = df
        self.shifted_df = df[seq_l - 1:].reset_index()
        self.seq_l = seq_l
        self.train = train
        
        self.train_tf = transforms.Compose([
            transforms.ColorJitter(brightness=0.05, contrast=0.1),
#             transforms.RandomCut(4)
        ])
    
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
            
            img_index = str(start_index - i).zfill(4)
            img = Image.open(self.data_dir + img_index + '.jpg')

            x = self.transform(img)
            X.append(x)
        
        X = torch.stack(X, dim=0)
         
        all_steer = np.clip(all_steer,-1.0, 1.0)
        
        steer_angles = torch.FloatTensor(all_steer[:-1])

        last_steer = torch.FloatTensor([all_steer[-1]])
        
        return X, last_steer, steer_angles


class TestDataset(Dataset):

    def __init__(self, data_dir, df):

        self.data_dir = data_dir
        
        self.df = df.reset_index()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.df)
     
    def __getitem__(self, index):
        
        start_index = self.df['index'].iloc[index]
        img_index = str(start_index).zfill(4)
        img = Image.open(self.data_dir + img_index + '.jpg')
        x = self.transform(img)
        
        last_steer = self.df.data.loc[index]
        
        return x, last_steer 
        
        
        
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