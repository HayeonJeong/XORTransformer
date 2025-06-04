import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
from tqdm import tqdm

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

class AttentionWeights(nn.Module):
    def __init__(self, D_k, weights, bias):
        super(AttentionWeights, self).__init__()
        self.D_k = D_k
        self.weights = weights
        self.bias = bias
        
        # Initialize attention weights
        self.W_q = nn.Linear(1, self.D_k, bias=False)
        self.W_k = nn.Linear(1, self.D_k, bias=False)
        self.W_v = nn.Linear(1, self.D_k, bias=False)
        
        if not self.weights:
            with torch.no_grad():
                ones = torch.ones(self.D_k, 1)
                self.W_q.weight.data.copy_(ones)
                self.W_k.weight.data.copy_(ones) 
                self.W_v.weight.data.copy_(ones)
            self.W_q.weight.requires_grad = False
            self.W_k.weight.requires_grad = False
            self.W_v.weight.requires_grad = False
        else:
            # Initialize weights with small random values
            nn.init.xavier_uniform_(self.W_q.weight)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.xavier_uniform_(self.W_v.weight)

        # Initialize bias
        if self.bias:
            self.b_q = nn.Parameter(torch.zeros(self.D_k))
            self.b_k = nn.Parameter(torch.zeros(self.D_k))
            self.b_v = nn.Parameter(torch.zeros(self.D_k))
        else:
            self.register_buffer("b_q", torch.zeros(self.D_k))
            self.register_buffer("b_k", torch.zeros(self.D_k))
            self.register_buffer("b_v", torch.zeros(self.D_k))

    def forward(self, x_nth):
        x_nth = x_nth.unsqueeze(0)

        # Apply query, key, value transformations
        q_l = self.W_q(x_nth)
        k_l = self.W_k(x_nth)
        v_l = self.W_v(x_nth)
        
        # Add bias if needed
        if self.bias:
            q_l = q_l + self.b_q
            k_l = k_l + self.b_k
            v_l = v_l + self.b_v
            
        return q_l, k_l, v_l

class XORTransformer(nn.Module):
    def __init__(self, conditions):
        super(XORTransformer, self).__init__()
        # All conditions
        self.D_k = conditions["D_k"]
        self.weights = conditions["weights"]
        self.bias = conditions["bias"]
        self.positional_encoding = conditions["positional_encoding"]
        self.softmax = conditions["softmax"]
        self.layer_norm = conditions["layer_norm"]

        # 2nd, 3rd condition: attention weights
        self.attention_weights = AttentionWeights(self.D_k, self.weights, self.bias)

        # 6th condition: Layer Normalization
        self.norm = nn.LayerNorm(self.D_k)

        # Linear layer for final output
        self.linear = nn.Linear(self.D_k, 1)

    def sinusoidal_positional_encoding(self, x_l, x_r):
        #  scalar sinusoidal encoding for each position
        pe_l = math.sin(0 * 1 / (10000 ** (0 / 1)))  # = sin(0) = 0.0
        pe_r = math.sin(1 * 1 / (10000 ** (0 / 1)))  # = sin(1 / 10000)

        return x_l + pe_l, x_r + pe_r

    def forward(self, x):
        x_l = x[0]
        x_r = x[1]

        # 1. positional encoding
        if self.positional_encoding:
            x_l, x_r = self.sinusoidal_positional_encoding(x_l, x_r)
        
        # 2. query, key, value computation
        q1, k1, v1 = self.attention_weights(x_l)
        q2, k2, v2 = self.attention_weights(x_r)

        # 3. attention score computation
        if self.softmax:
            a11 = F.softmax((q1 * k1) / math.sqrt(self.D_k), dim=-1)
            a12 = F.softmax((q1 * k2) / math.sqrt(self.D_k), dim=-1)
            a21 = F.softmax((q2 * k1) / math.sqrt(self.D_k), dim=-1)
            a22 = F.softmax((q2 * k2) / math.sqrt(self.D_k), dim=-1)
        else:
            a11 = (q1 * k1) / math.sqrt(self.D_k)
            a12 = (q1 * k2) / math.sqrt(self.D_k)
            a21 = (q2 * k1) / math.sqrt(self.D_k)
            a22 = (q2 * k2) / math.sqrt(self.D_k)

        # 4. updating
        y1 = a11 * v1 + a12 * v2
        y2 = a21 * v1 + a22 * v2

        # 5. layer normalization
        if self.layer_norm:
            y1 = self.norm(y1)
            y2 = self.norm(y2)

        # 6. linear layer
        y1 = self.linear(y1)
        y2 = self.linear(y2)

        # 7. sigmoid for probability
        prob = torch.sigmoid(y1)
        
        # 8. output probability directly
        return prob, int(prob > 0.5)
