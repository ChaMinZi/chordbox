import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torchmetrics import Accuracy

from decoder import *
from encoder import *


class ChordTransformer(pl.LightningModule):
    def __init__(self, n_classes:int=26, seg_width:int=21, freq_bandwidth:int=24, lambda1:int=3, lambda2:int=1, is_rpe=True, dropout_rate=0., device=None):
        super(ChordTransformer, self).__init__()
        
        self.dev = device
        
        self.encoder = ChordEncoder(seg_width, freq_bandwidth, is_rpe=is_rpe, dropout_rate=dropout_rate, device=device)
        self.decoder = ChordDecoder(n_classes, seg_width, freq_bandwidth, is_rpe=is_rpe, dropout_rate=dropout_rate, device=device)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.train_losses = []
        self.train_accrs = []
        self.val_losses = []
        self.val_accrs = []


    def forward(self, x:Tensor):
        r_enc, logits_enc, p_enc, o_enc = self.encoder(x)

#         B,T,_ = x.shape
#         r_enc = torch.randn(B,T,512, device=self.dev)
#         p_enc = torch.nn.Sigmoid()(torch.randn(B,T, device=self.dev))
#         o_enc = torch.where(p_enc<0.5, 0, 1)
        
        p_dec, o_dec = self.decoder(x, o_enc, r_enc)

#         B,T = p_enc.shape
#         p_dec = torch.randn(B,T,26, device=self.dev)
#         o_dec = torch.randn_like(p_enc, device=self.dev)
        
        out_dict = {'logits_enc':logits_enc, 'p_enc':p_enc, 'o_enc':o_enc, 'p_dec':p_dec, 'o_dec':o_dec}

        return out_dict
        
        
    def criterion(self, logits_enc, true_o_enc, p_dec, true_o_dec):
        '''
        p_enc shape (B, T)
        true_o_enc shape (B, T)
        p_dec shape (B, T, n_classes)
        true_o_dec shape (B, T)
        '''
#         print(p_dec.dtype)
#         print(true_o_dec.dtype)
#         enc_loss = nn.BCELoss()(p_enc, true_o_enc)
        enc_loss = nn.BCEWithLogitsLoss()(logits_enc, true_o_enc)
        dec_loss = nn.CrossEntropyLoss()(p_dec.permute(0,2,1), true_o_dec)  ##p_dec shape (B, T, n_classes) => (B, n_classes, T) for nn.CrossEntropyLoss()
        loss = self.lambda1*enc_loss + self.lambda2*dec_loss
        return loss
        
    
    def training_step(self, train_batch, batch_idx):
        x_train = train_batch['x_train']
        true_o_enc = train_batch['y_cc_train'].to(dtype=torch.float32)
        true_o_dec = train_batch['y_train'].to(dtype=torch.long)
        
        
        out_dict = self.forward(x_train)
        logits_enc = out_dict['logits_enc']#
        p_enc = out_dict['p_enc']
        o_enc = out_dict['o_enc']
        p_dec = out_dict['p_dec']
        o_dec = out_dict['o_dec']
        
        loss = self.criterion(logits_enc, true_o_enc, p_dec, true_o_dec)
        accr = Accuracy()(o_dec.cpu(), true_o_dec.cpu())
        pbar = {'train_acc':accr}

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_accr', accr, on_step=False, on_epoch=True)
        
        self.train_losses.append(loss.detach().cpu())
        self.train_accrs.append(accr.detach().cpu())
        
        return {'loss':loss, 'progress_bar':pbar}
        
    
    def validation_step(self, val_batch, batch_idx):
        x_valid = val_batch['x_valid']
        true_o_enc = val_batch['y_cc_valid'].to(dtype=torch.float32)
        true_o_dec = val_batch['y_valid'].to(dtype=torch.long)
        
        out_dict = self.forward(x_valid)
        logits_enc = out_dict['logits_enc']#
        p_enc = out_dict['p_enc']
        o_enc = out_dict['o_enc']
        p_dec = out_dict['p_dec']
        o_dec = out_dict['o_dec']
        
        loss = self.criterion(logits_enc, true_o_enc, p_dec, true_o_dec)
        accr = Accuracy()(o_dec.cpu(), true_o_dec.cpu())
        pbar = {'val_acc':accr}
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_accr', accr, on_step=False, on_epoch=True)

        self.val_losses.append(loss.detach().cpu())
        self.val_accrs.append(accr.detach().cpu())
 
        return {'loss':loss, 'progress_bar':pbar}
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
        lr_dict = {"scheduler":scheduler, "monitor": "val_loss"}
        
        return [optimizer], [lr_dict]
