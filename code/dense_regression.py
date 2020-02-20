import torch
from torch import nn
import math


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Intermediate(nn.Module):
    def __init__(self,hidden_size,layer_norm_eps,hidden_dropout_prob):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.intermediate_act_fn = gelu
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class DenseNN(nn.Module):
    def __init__(self,nlayer,hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob):
        layers=[]
        for i in range(nlayer)
            layers.append(Intermediate(hidden_size,intermediate_size,layer_norm_eps,hidden_dropout_prob))
        self.layers=nn.Sequential(*layers)

        self.start=nn.Linear(input_size, hidden_size)

    def forward(self,input):

