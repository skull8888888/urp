import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict
from typing import List
import timm

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        x = x + self.pe[:x.size(0), :]
                  
        return self.dropout(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
class ResidualEncoderAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.d_model = d_model
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        
        self.attn_mask = attn_mask
        
        self.initialize_parameters()
        
        
    def initialize_parameters(self):

        proj_std = (self.d_model ** -0.5) * ((2 * 1) ** -0.5)
        attn_std = self.d_model ** -0.5
        fc_std = (2 * self.d_model) ** -0.5

        nn.init.normal_(self.attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.attn.out_proj.weight, std=proj_std)

        nn.init.normal_(self.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.mlp.c_proj.weight, std=proj_std)
        
    
    def attention(self, x):

        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    
    def forward(self, x):    
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class ResidualDecoderAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn_mask = attn_mask
        
        self.d_model = d_model
        
        self.attn_1 = nn.MultiheadAttention(d_model, n_head)
        self.attn_2 = nn.MultiheadAttention(d_model, n_head)
        
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        
        self.initialize_parameters()

        
    def initialize_parameters(self):

        proj_std = (self.d_model ** -0.5) * (2 ** -0.5)
        attn_std = self.d_model ** -0.5
        fc_std = (2 * self.d_model) ** -0.5

        nn.init.normal_(self.attn_1.in_proj_weight, std=attn_std)
        nn.init.normal_(self.attn_1.out_proj.weight, std=proj_std)

        nn.init.normal_(self.attn_2.in_proj_weight, std=attn_std)
        nn.init.normal_(self.attn_2.out_proj.weight, std=proj_std)

        nn.init.normal_(self.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.mlp.c_proj.weight, std=proj_std)
        
            
    def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor):
                
        x = self.ln_1(dec_input)
        
        x = dec_input + self.attn_1(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        
        x_norm = self.ln_2(x)
        
        x = x + self.attn_2(x_norm, enc_output, enc_output, need_weights=False, attn_mask=self.attn_mask)[0]
        x = x + self.mlp(self.ln_3(x))

        return x

class Encoder(nn.Module):
    
    def __init__(self, d_model: int, n_head: int, layers: int, seq_l: int, attn_mask=None, p:float = 0.1):
        super().__init__()
        
        self.p = p
        self.seq_l = seq_l
        
        self.pos_emb = PositionalEncoding(d_model, max_len=seq_l)
            
        nn.init.normal_(self.embedding.weight, std=0.02)
        
        self.blocks = nn.Sequential(*[ResidualEncoderAttentionBlock(d_model, n_head, attn_mask=attn_mask) for _ in range(layers)])
    
        self.ln_post = nn.LayerNorm(d_model)
        
      
    def forward(self, x: torch.Tensor):
        
        x = self.pos_emb(x)
        x = self.blocks(x)
        
        x = self.ln_post(x)
        
        return x


class Decoder(nn.Module):
    
    # needed for torch.jit.script conversion
    __annotations__ = {
        "p" : float
    }
    
    def __init__(self, d_model: int, n_head: int, layers: int, seq_l: int, attn_mask=None, p:float = 0.1):
        super().__init__()
        
        self.p = p
        self.seq_l = seq_l
        
        self.pos_emb = PositionalEncoding(d_model, max_len=seq_l)
             
        self.embedding = nn.Linear(1, d_model) 
        
#         nn.init.normal_(self.embedding.weight, std=0.02)
            
        self.blocks = [ResidualDecoderAttentionBlock(d_model, n_head, attn_mask=attn_mask) for _ in range(layers)]
    
        self.ln_post = nn.LayerNorm(d_model)
        
        self.fc = nn.Linear(d_model, 1)

        
    def forward(self, steer_angles: torch.Tensor, enc_output: torch.Tensor):
        '''
        img: (batch_size, seq_l, d_model)
        steer_tokens: (batch_size, seq_l-1, d_model)
        '''
        
        # convert angles to tokens
        
        steer_tokens = self.embedding(steer_angles)
        
        x = self.pos_emb(steer_tokens)
        
        for block in self.blocks:
            
            x = block(x, enc_output)
        
        x = x.transpose(0,1) # (batch_size, 1, d_model)
        
        x = self.ln_post(x)
        
        x = F.dropout(x, training=self.training, p=self.p)

        x = self.fc(x).flatten()

        return x
