import numpy as np
import torch
from torch import nn, Tensor
from common_classes import *
from relative_pe import *


class DecoderInputEmbedding(nn.Module):
    def __init__(self, seg_width:int, freq_bandwidth:int, embedding_dim:int, is_relative:bool=True, dropout_rate:float=0., device=None):
        super(DecoderInputEmbedding, self).__init__()

        self.device = device
        self.SW = seg_width
        self.FB = freq_bandwidth
        self.dropout_rate = dropout_rate

        self.is_relative = is_relative
        if is_relative:
            self.pe = RelativeGlobalAttention(d_model=self.SW, num_heads=3, dropout=dropout_rate, device=device)
        else:
            self.pe = MHA(input_dim=self.SW, num_heads=1, dropout_rate=dropout_rate, device=device)
            
        self.ffn = FFN(input_dim=self.SW, hidden_dims=[self.SW*4, self.SW], dropout_rate=dropout_rate, device=device)
        self.to_embedding_size = nn.Linear(in_features=self.SW*self.FB, out_features=embedding_dim, device=device)


    def _segmentwise_mean(self, data:Tensor, index:Tensor):
        B, T, F = data.shape
        ## index shape (B, T)
        counts = torch.zeros_like(index, device=self.device).scatter_add_(dim=1, index=index, src=torch.ones_like(index, device=self.device)).clamp(min=1)
        segmentwise_mean = torch.zeros_like(data, device=self.device)
#         time_id = 0
        for b in range(B):
            
            cnt_id = 0
#             i = 0
            for cnt in counts[b]:
                if cnt_id>=T:
                    break

                block_mean = data[b, cnt_id:cnt_id+cnt, :].mean(dim=0)
                segmentwise_mean[b, cnt_id] = block_mean
                cnt_id += cnt
#                 i += 1
            
            
#             max_idx = index[b].max()
#             for i, idx in enumerate(index[b]):  ## sum
#                 if i>max_idx:
#                     break
                    
#                 time_id = idx
#                 segmentwise_mean[b,time_id,:] += data[b,i,:]
                
# #                 if id==time_id:
# #                     segmentwise_mean[b,time_id,:] += data[b,i,:]
# #                 else:
# #                     time_id = id
# #                     segmentwise_mean[b,time_id,:] += data[b,i,:]    

#             for i, cnt in enumerate(counts[b]):  ## mean (sum/counts)
#                 segmentwise_mean[b, i, :] /= cnt

        return segmentwise_mean


    def _regionalize(self, e_dec:Tensor, o_enc:Tensor):
        assert o_enc.dim()==2 ##(B, T)
        
        block_ids = torch.cumsum(o_enc, dim=1)
        for b in block_ids:
            if b[0]==1:
                b -= 1

        num_blocks = torch.max(block_ids, dim=1).values + 1
        block_means = self._segmentwise_mean(e_dec, block_ids)

        return block_means, block_ids, num_blocks


    def forward(self, x:Tensor, o_enc:Tensor, r_enc:Tensor) -> Tensor:
        B, T, F = x.shape
        B2 = B*T
        if F != self.SW*self.FB:
            raise ValueError(f'Feature dimension({F}) does not equal segment_width({self.SW}) * frequency_bandwidth({self.FB})')

        ## reshape !!!!!!! Different from the orignal tf1 code !!!!!!! (in terms of reshaping)
        x_reshaped = x.reshape(B2, self.SW, self.FB)  ## (B, T, F) => (B*T, SW, FB)
        x_reshaped = x_reshaped.permute(0,2,1)  ## shape (B*T, SW, FB) => (B*T, FB, SW)

        ## add sinusoidal encoding
        if not self.is_relative:
            x_reshaped += sinusoidal_encoding(*(x_reshaped.shape), device=self.device) * 0.01 + 0.01
         
        emb = self.pe(x_reshaped, x_reshaped, x_reshaped)  ## time_element: input, output : shape (B*T, FB, SW)
        emb = self.ffn(emb)  ## Linear transformation over the last dimension : shape (B*T, FB, SW)

        ## restore shape
        emb = emb.permute(0,2,1)  ## shape (B*T, FB, SW) => (B*T, SW, FB)
        emb = emb.reshape(B, T, F)

        ## to embedding_dim
        emb = nn.Dropout(self.dropout_rate)(emb)
        emb = self.to_embedding_size(emb)  ## shape (B, T, embedding_dim)
        emb = nn.LayerNorm(emb.shape, eps=1e-8, device=self.device)(emb)
        
        ## Regionalization
        block_means, block_ids, num_blocks = self._regionalize(emb, o_enc)  ## block_means:shape(B, T, embedding_dim), block_ids:shape(B,T), num_blocks:shape(B)  
        
#         block_means = torch.randn(B,T, 512, device=self.device)
#         block_ids = torch.randint(B,T, device=self.device)
#         num_blocks = torch.randint(B)

        out = block_means + r_enc + emb  ## shape (B, T, embedding_dim)

        ## Input Embedding + PE
        if not self.is_relative:
            out += sinusoidal_encoding(*(out.shape), device=self.device)
        out = nn.Dropout(self.dropout_rate)(out)

        return out
        
        

class ChordDecoderLayer(nn.Module):
    def __init__(self, input_dim:int, num_heads:int, hidden_dims:list, dropout_rate:float=0., device=None):
        super(ChordDecoderLayer, self).__init__()

        self.dropout_rate = dropout_rate
        self.self_mha = MHA(input_dim, num_heads, dropout_rate, device)
        self.enc_dec_mha = MHA(input_dim, num_heads, dropout_rate, device)
        self.ffn = FFN(input_dim, hidden_dims, dropout_rate, device)


    def forward(self, x:Tensor, r_enc:Tensor) -> Tensor:
        '''
        x shape (B,T,embedding_dim), r_enc shape (B,T,embedding_dim)
        '''
        z = self.self_mha(query=x, key=x, value=x)
        out = self.enc_dec_mha(query=z, key=r_enc, value=r_enc)
        out = self.ffn(out)

        return out
        
        
        
class ChordDecoder(nn.Module):
    def __init__(self, 
                 n_classes:int,
                 seg_width:int, 
                 freq_bandwidth:int, 
                 input_embedding_dim:int=512, 
                 num_heads:int=8,
                 n_layers:int=2,
                 dec_weights:list=[1,1],
                 dropout_rate:float=0.,
                 is_rpe:bool=False,
                 device=None):
        
        super(ChordDecoder, self).__init__()
        
        self.n_layers = n_layers
        self.input_embedder = DecoderInputEmbedding(seg_width, freq_bandwidth, input_embedding_dim, dropout_rate=dropout_rate, is_relative=is_rpe, device=device)
            
        decoder_dict = {}
        for l in range(n_layers):
            decoder_dict[f'decoder_layer{l}'] = ChordDecoderLayer(input_embedding_dim, num_heads, hidden_dims=[input_embedding_dim*4, input_embedding_dim], dropout_rate=dropout_rate, device=device)
        self.decoder_dict = nn.ModuleDict(decoder_dict)

        dec_weights = torch.tensor(dec_weights, dtype=torch.float32)
        self.softmax_dec_weights = nn.Softmax(dim=0)(dec_weights)
        self.map_to_logits = nn.Linear(in_features=input_embedding_dim, out_features=n_classes, device=device)

        # self.slope = 1  ## annealing slope -> as it increases, the logits are pushed to either extremes, binarizing the change probabilities


    def forward(self, x:Tensor, o_enc:Tensor, r_enc:Tensor):
        emb = self.input_embedder(x, o_enc, r_enc)
        weighted_dec_sum = torch.zeros_like(emb)
        for l in range(self.n_layers):
            emb = self.decoder_dict[f'decoder_layer{l}'](emb, r_enc)
            weighted_dec_sum += (self.softmax_dec_weights[l] * emb)

        logits = self.map_to_logits(weighted_dec_sum)  ## shape (B, T, n_classes)
        probs = nn.Softmax(dim=2)(logits) ## shape (B, T, n_classes)
        preds = torch.argmax(probs, dim=2) ## shape (B, T)

        return probs, preds
        