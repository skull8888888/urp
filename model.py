import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning.metrics.functional as plm
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import math
from collections import OrderedDict
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
#         print(x.shape, self.pe[:x.size(0), :].shape, self.pe.shape)

        assert x.size(0) <= self.pe.size(0)
        
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

        if self.attn_mask != None:
            attn_mask = self.attn_mask.type_as(x)
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

        return self.attn(x, x, x, need_weights=False)[0]

    
    def forward(self, x):    
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class ResidualDecoderAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

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

        proj_std = (self.d_model ** -0.5) * ((2 * 1) ** -0.5)
        attn_std = self.d_model ** -0.5
        fc_std = (2 * self.d_model) ** -0.5

        nn.init.normal_(self.attn_1.in_proj_weight, std=attn_std)
        nn.init.normal_(self.attn_1.out_proj.weight, std=proj_std)

        nn.init.normal_(self.attn_2.in_proj_weight, std=attn_std)
        nn.init.normal_(self.attn_2.out_proj.weight, std=proj_std)

        nn.init.normal_(self.mlp.c_fc.weight, std=fc_std)
        nn.init.normal_(self.mlp.c_proj.weight, std=proj_std)
        
            
    def forward(self, dec_input, enc_output, mask=None, pad_mask=None):
                
        x = self.ln_1(dec_input)
        
        x = dec_input[-1:,:,:] + self.attn_1(x[-1:,:,:], x, x, need_weights=False)[0]
        
        x_norm = self.ln_2(x)
        
        x = x + self.attn_2(x_norm, enc_output, enc_output, need_weights=False)[0]
        x = x + self.mlp(self.ln_3(x))

        return x
    

class Backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        backbone = timm.create_model(backbone, pretrained=True) 
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
                
    def forward(self, x):        
        
        x = self.backbone(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_l: int, stride: int, p=0.1):
        super().__init__()
        
        self.p = p
        self.seq_l = seq_l
        
        self.encoder_pos_emb = PositionalEncoding(d_model, max_len=seq_l)
        self.decoder_pos_emb = PositionalEncoding(d_model, max_len=seq_l)
        
#         bins = torch.arange(-1-stride/2, 1+stride/2, stride)
        bins = torch.Tensor([-1.1, -0.5, 0.5, 1.1])
        self.register_buffer('bins', bins)
    
        self.embedding = nn.Embedding(len(self.bins), d_model) # + 1 because of REG token
        
        
        nn.init.normal_(self.embedding.weight, std=0.02)
        
#         mask = (torch.triu(torch.ones(seq_l, seq_l)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         mask = None
    
        self.encoder = ResidualEncoderAttentionBlock(d_model, n_head, attn_mask=None)
        self.decoder = ResidualDecoderAttentionBlock(d_model, n_head)

        self.ln_post = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        
        
    def encode_frames(self, x):
                
        # x: (batch_size, L, d_model)
        
        x = x.transpose(0,1) # (L, batch_size, d_model)    
        
        x = self.encoder_pos_emb(x)
        x = self.encoder(x)
        
        return x

    def encode_steer(self, steer_token):

        x = self.embedding(steer_token) #(batch_size, L-1, d_model)
        x = x.transpose(0,1) # (L-1, batch_size, d_model)    
        x = self.decoder_pos_emb(x)
        
        return x
    
    def tokenize(self, steer_angles: torch.Tensor):
        '''
        steer_angles: (batch_size, seq_l - 1) 
        '''
        
        batch_size = steer_angles.size(0)
        
        steer_tokens = torch.bucketize(steer_angles, self.bins)
        
        assert steer_tokens.max().item() <= len(self.bins) 

        reg_token = torch.zeros(batch_size,1).type_as(steer_angles).long() # 0 REG token

        tokens = torch.cat([steer_tokens, reg_token], dim=-1)
        
        return tokens
        
        
        
    def forward(self, frames: torch.Tensor, steer_angles: torch.Tensor):
        
        # img: (batch_size, seq_l, d_model)
        # steer_token: (batch_size, seq_l-1, d_model)
        
        assert frames.size(1) == self.seq_l
        assert steer_angles.size(1) == self.seq_l - 1
        
        steer_tokens = self.tokenize(steer_angles) # needed for pytorch lightning
        
        frames_enc = self.encode_frames(frames) # (L, batch_size, d_model)
        steer_enc = self.encode_steer(steer_tokens) # (L, batch_size, d_model) because added REG token

#         print(steer_enc.shape)
        assert steer_enc.size(0) == self.seq_l
        
        x = self.decoder(steer_enc, frames_enc)
        x = x.transpose(0,1) # (batch_size, 1, d_model)
        
        x = self.ln_post(x)
        
        x = F.dropout(x, training=self.training, p=self.p)

        x = self.fc(x)

        return x
        
class Model(pl.LightningModule):
    
    def __init__(self, cfg, train_loader_len=0):
        super(Model, self).__init__()
        
        self.train_loader_len = train_loader_len
        
        self.save_hyperparameters(cfg)
        
        self.backbone = Backbone(self.hparams.backbone)
        self.decoder = Decoder(self.hparams.d_model, self.hparams.n_head, self.hparams.seq_l, self.hparams.stride, p=self.hparams.p)
        
        self.frames_cache = []
        self.angles_cache = []
                    
    
    def training_step(self, batch, batch_nb):
        
        imgs, last_steer, steer_angles = batch
        
        x = []
        
        for i in range(imgs.size(1)):
            x.append(self.backbone(imgs[:,i,:,:,:]))
        x = torch.stack(x, dim=1)
        
        y_hat = self.decoder(x, steer_angles).flatten()
        
        loss = F.l1_loss(y_hat, last_steer.flatten())
           
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, prog_bar=True)
        
        return loss
           
    def validation_step(self, batch, batch_nb):
        
        img, steer = batch 
                    
        """
        img: (1, 3, w, h)
        """
        x = self.backbone(img)
        self.frames_cache.append(x)
        
        if len(self.frames_cache) < self.hparams.seq_l:
            
            self.angles_cache.append(torch.FloatTensor([0]).type_as(img))
            
            return {'pred': 0, 'target': 0}
        
        x = torch.stack(self.frames_cache, dim=1)
        
        steer_angles = torch.stack(self.angles_cache).T
#         print(steer_angles.shape)
        
        y_hat = self.decoder(x, steer_angles).detach().flatten()
        y_hat = torch.clip(y_hat, -1.0, 1.0)
        
        self.angles_cache.append(y_hat)
        self.angles_cache = self.angles_cache[1:]
        
        self.frames_cache = self.frames_cache[1:]

        return {'pred': y_hat.cpu().item(), 'target': steer.detach().cpu().item()}
    
    def validation_epoch_end(self, outputs):
        
        pred = np.array([x['pred'] for x in outputs])
        target = np.array([x['target'] for x in outputs])
        
        l1 = np.abs(target - pred).mean()
        
        self.log('val_loss', l1, prog_bar=True)

        self.frames_cache = []
        self.angles_cache = []                    
    
#     def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
                
#         optimizer.step(closure=second_order_closure)

#         self.scheduler.step(current_epoch + batch_nb / self.train_loader_len)
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            div_factor=self.hparams.div_factor,
            final_div_factor=self.hparams.final_div_factor,
            pct_start=self.hparams.pct_start,
            max_lr=self.hparams.lr,
            cycle_momentum=False,
            anneal_strategy=self.hparams.anneal,
            steps_per_epoch=self.train_loader_len, 
            epochs=self.hparams.epochs)
        
        lr_scheduler = {'scheduler': self.scheduler,
                        'interval':'step'}
    
        return [optimizer], [lr_scheduler]
        