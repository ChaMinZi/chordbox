import numpy as np
import torch
from torch import nn, Tensor
from common_classes import *
from relative_pe import *


class EncoderInputEmbedding(nn.Module):
    def __init__(self, seg_width:int, freq_bandwidth:int, embedding_dim:int, is_relative:bool=True, dropout_rate:float=0., device=None):
        '''
        Where X2:= X.reshape(B*T, SW, FB), Embed(X2):= FFN(MHA(X2)).reshape(B,T,F)
        
        InputEmbedding(X) = Linear(Embed(X2+PE2))+PE

        Input: shape (B, T=100, F=504)
        Output: shape (B, T, embedding_dim)
        '''
        super(EncoderInputEmbedding, self).__init__()

        self.device = device
        self.SW = seg_width
        self.FB = freq_bandwidth
        self.dropout_rate = dropout_rate

        self.is_relative = is_relative
        if is_relative:
            self.pe = RelativeGlobalAttention(d_model=self.FB, num_heads=3, dropout=dropout_rate, device=device)
        else:
            self.pe = MHA(input_dim=self.FB, num_heads=2, dropout_rate=dropout_rate, device=device)
            
        self.ffn = FFN(input_dim=self.FB, hidden_dims=[self.FB*4, self.FB], dropout_rate=dropout_rate, device=device)            
        self.to_embedding_size = nn.Linear(in_features=self.SW*self.FB, out_features=embedding_dim, device=device)
        

    def forward(self, x:Tensor) -> Tensor:
        B, T, F = x.shape
            
        B2 = B*T
        if F != self.SW * self.FB:
            raise ValueError(f'Feature dimension({F}) does not equal segment_width({self.SW}) * frequency_bandwidth({self.FB})')

        ## reshape !!!!!!! Different from the orignal tf1 code !!!!!!! (in terms of reshaping)
        x_reshaped = x.reshape(B2, self.SW, self.FB)  ## (B, T, F) => (B*T, SW, FB)
        
        ## embed
        if not self.is_relative:
            x_reshaped += sinusoidal_encoding(*(x_reshaped.shape), device=self.device) * 0.01 + 0.01

        emb = self.pe(x_reshaped, x_reshaped, x_reshaped)  ## time_element: input, output : shape (B*T, SW, FB)
        out = self.ffn(emb)  ## Linear transformation over the last dimension : shape (B*T, SW, FB)

        ## restore shape
        out = out.reshape(B, T, F)  

        ## Input Embedding
        out = nn.Dropout(self.dropout_rate)(out)
        out = self.to_embedding_size(out)  ## shape (B, T, embedding_dim)
        out = nn.LayerNorm(out.shape, eps=1e-8, device=self.device)(out)
        
        ## Input Embedding + PE
        if not self.is_relative:
            out += sinusoidal_encoding(*(out.shape), device=self.device)
        out = nn.Dropout(self.dropout_rate)(out)

        return out


class ChordEncoderLayer(nn.Module):
    def __init__(self, input_dim:int, num_heads:int, hidden_dims:list, dropout_rate:float=0., device=None):
        super(ChordEncoderLayer, self).__init__()

        self.dropout_rate = dropout_rate
        self.mha = MHA(input_dim, num_heads, dropout_rate, device)
        self.ffn = FFN(input_dim, hidden_dims, dropout_rate, device)

    def forward(self, x:Tensor) -> Tensor:
        out = self.mha(x, x, x)
        out = self.ffn(out)

        return out
        
        
        
class ChordEncoder(nn.Module):
    def __init__(self, 
                 seg_width:int, 
                 freq_bandwidth:int, 
                 input_embedding_dim:int=512, 
                 num_heads:int=8,
                 n_layers:int=2,
                 enc_weights:list=[1,1],
                 dropout_rate:float=0.,
                 is_rpe:bool=False,
                 device=None):
        
        super(ChordEncoder, self).__init__()

        self.n_layers = n_layers    
        self.input_embedder = EncoderInputEmbedding(seg_width, freq_bandwidth, input_embedding_dim, is_relative=is_rpe, dropout_rate=dropout_rate, device=device)

        encoder_dict = {}
        for l in range(n_layers):
            encoder_dict[f'encoder_layer{l}'] = ChordEncoderLayer(input_embedding_dim, num_heads, hidden_dims=[input_embedding_dim*4, input_embedding_dim], dropout_rate=dropout_rate, device=device)
        self.encoder_dict = nn.ModuleDict(encoder_dict)

        enc_weights = torch.tensor(enc_weights, dtype=torch.float32)
        self.softmax_enc_weights = nn.Softmax(dim=0)(enc_weights)
        self.map_to_logits = nn.Linear(in_features=input_embedding_dim, out_features=1, device=device)

        self.slope = 1  ## annealing slope -> as it increases, the logits are pushed to either extremes, binarizing the change probabilities

    def forward(self, x:Tensor) -> Tensor:
        '''
        Input shape (B, T, F)
        '''        
        emb = self.input_embedder(x)   ## shape (B, T, embedding_dim)
        weighted_enc_sum = torch.zeros_like(emb)
        for l in range(self.n_layers):
            emb = self.encoder_dict[f'encoder_layer{l}'](emb)
            weighted_enc_sum += (self.softmax_enc_weights[l] * emb)
        
        chord_change_logits = self.map_to_logits(weighted_enc_sum) ## (B, T, embedding_dim) => (B, T, 1)
        chord_change_logits = chord_change_logits.squeeze(2)  ## shape (B, T)

#         chord_change_probs = nn.Sigmoid()(self.slope*chord_change_logits)
        chord_change_probs = SlopeSigmoid(slope=2)(self.slope * chord_change_logits) #slopesigmoid's slope and self.slope are not the same
        chord_change_preds = torch.where(chord_change_probs<0.5, 0, 1)

        return weighted_enc_sum, chord_change_logits, chord_change_probs, chord_change_preds  ## r_enc, logits_enc, p_enc, o_enc

