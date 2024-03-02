import numpy as np
import torch
from torch import nn, Tensor

class SlopeSigmoid(nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope
        
    def forward(self, x):
        response = 1/(1+self.slope*torch.exp(-x))
        return response

    
class FFN(nn.Module):
    '''
    FFN(X) = LayerNorm( Linear(ReLU(Linear(X))) + X)
    Where the Linear layer linearly sums over 'Feature Bandwidth' (=24), where X is of shape (B*T, SW, FB)
    '''
    def __init__(self, input_dim:int, hidden_dims:list=[2048, 512], dropout_rate:float=0., device=None):
        '''
        hidden_dims : list of length 2, one for each Linear layer
        '''
        super(FFN, self).__init__()
        
        if len(hidden_dims)!=2:
            raise ValueError('hidden_dims should be a list of length 2')

        self.device = device
        self.dropout_rate = dropout_rate
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0], device=device),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1], device=device),
        )

    def forward(self, x:Tensor) -> Tensor:
        out = self.layers(x)
        out = nn.Dropout(self.dropout_rate)(out)
        out += x
        out = nn.LayerNorm(out.shape, eps=1e-8, device=self.device)(out)

        return out



class MHA(nn.Module):
    '''
    MHA(X) = LayerNorm(MultiHeadAttention(X)+X)
    '''
    def __init__(self, input_dim:int, num_heads:int=8, dropout_rate:float=0., device=None):
        super(MHA, self).__init__()
        
        self.device = device
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True, device=device)

    def forward(self, query:Tensor, key:Tensor, value:Tensor) -> Tensor:
        out, weights = self.mha(query, key, value, need_weights=False)
        out += query
        out = nn.LayerNorm(out.shape, eps=1e-8, device=self.device)(out)
        
        return out
        
        

def sinusoidal_encoding(batch_size:int, time_steps:int, dims:int, device=None) -> Tensor:
    '''
    Sinusoidal encoding
    '''
    # batch_size, time_steps, dims  ## B*T, 21, 24

    position_ind = torch.tile(torch.arange(0,time_steps).unsqueeze(0), [batch_size, 1])

    position_enc = np.array([[pos / np.power(10000, 2.*i/dims) for i in range(dims)] for pos in range(time_steps)], dtype=np.float32)  ## shape (time_steps, dims) = (21, 24)
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2]) # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) # dim 2i+1
    position_enc = torch.tensor(position_enc)

    lookup = nn.Embedding.from_pretrained(position_enc, freeze=True)
    outputs = lookup(position_ind)  ## shape (batch_size, time_steps, dims)
    outputs = outputs.to(device)
    return outputs
