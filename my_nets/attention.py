import torch
import torch.nn as nn
from torchvision import models
import torch.utils.checkpoint

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob       = 1 - drop_prob
    shape           = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor   = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() 
    output          = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


#--------------------------------------------------------------------------------------------------------------------#
#   Attention机制
#   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
#   然后利用 查询向量query 点乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
#   然后利用 score 点乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
#--------------------------------------------------------------------------------------------------------------------#
class Attention(nn.Module):
    def __init__(self, spatial=[32, 32], num_features=512, num_heads=2, qkv_bias=False, drop_rate=0.05):
        super().__init__()
        self.spatial = spatial
        num_patches = spatial[0] * spatial[1]
        self.pos_embed      = nn.Parameter(torch.zeros(1, num_patches, num_features))
        self.pos_drop       = nn.Dropout(p=drop_rate)
        
        self.num_heads  = num_heads
        self.scale      = (num_features // num_heads) ** -0.5

        self.qkv        = nn.Linear(num_features, num_features * 3, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(drop_rate)
        self.proj       = nn.Linear(num_features, num_features)
        self.proj_drop  = nn.Dropout(drop_rate)

        self.norm_layer = nn.LayerNorm(normalized_shape=num_features, eps=1e-6)
        self.drop_path  = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
    
    def forward_attention(self, x):
        B, N, C     = x.shape
        qkv         = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v     = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward(self, x):
        B, C, _, _ = x.shape

        # flatten
        x = x.flatten(2).transpose(1, 2)
        
        # position encoding
        x = self.pos_drop(x + self.pos_embed)

        # attention
        x = x + self.drop_path(self.forward_attention(self.norm_layer(x)))

        x = x.reshape(B, C, *self.spatial)
        return x
