import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum
import torch.nn.init as init
from torchsummary import summary

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out

class CTAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross token attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x
        
# cross token attention transformer

class CT_Transformer(nn.Module):
    def __init__(self, h_dim, l_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(h_dim, l_dim, LayerNormalize(l_dim, CTAttention(l_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(l_dim, h_dim, LayerNormalize(h_dim, CTAttention(h_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, h_tokens, l_tokens):
        (h_cls, h_patch_tokens), (l_cls, l_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (h_tokens, l_tokens))

        for h_attend_lg, l_attend_h in self.layers:
            h_cls = h_attend_lg(h_cls, context = l_patch_tokens, kv_include_self = True) + h_cls
            l_cls = l_attend_h(l_cls, context = h_patch_tokens, kv_include_self = True) + l_cls

        h_tokens = torch.cat((h_cls, h_patch_tokens), dim = 1)
        l_tokens = torch.cat((l_cls, l_patch_tokens), dim = 1)
        return h_tokens, l_tokens

# fusion encoder

class FusionEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        h_dim,
        l_dim,
        h_enc_params,
        l_enc_params,
        ct_attn_heads,
        ct_attn_depth,
        ct_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = h_dim, dropout = dropout, **h_enc_params),
                Transformer(dim = l_dim, dropout = dropout, **l_enc_params),
                CT_Transformer(h_dim = h_dim, l_dim = l_dim, depth = ct_attn_depth, heads = ct_attn_heads, dim_head = ct_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, h_tokens, l_tokens):
        for h_enc, l_enc, cross_attend in self.layers:
            h_tokens, l_tokens = h_enc(h_tokens), l_enc(l_tokens)
            h_tokens, l_tokens = cross_attend(h_tokens, l_tokens)

        return h_tokens, l_tokens
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

BATCH_SIZE_TRAIN = 1
NUM_CLASS = 6

class HCTnet(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        num_classes=NUM_CLASS, 
        num_tokens=4, 
        dim = 64,
        heads=8,
        mlp_dim=8,
        h_dim = 64,
        l_dim = 64,
        depth = 1,
        dropout = 0.1, 
        emb_dropout = 0.1,
        h_enc_depth = 1,
        h_enc_heads = 8,
        h_enc_mlp_dim = 8,
        # h_enc_dim_head = 64,
        l_enc_depth = 1,
        l_enc_heads = 8,
        l_enc_mlp_dim = 8,
        # l_enc_dim_head = 64,
        ct_attn_depth = 1,
        ct_attn_heads = 8,
        ct_attn_dim_head = 64,
    
    ):
        super(HCTnet, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2d_features2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens+1, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.fusion_encoder = FusionEncoder(
            depth = depth,
            h_dim = h_dim,
            l_dim = l_dim,
            ct_attn_heads = ct_attn_heads,
            ct_attn_dim_head = ct_attn_dim_head,
            ct_attn_depth = ct_attn_depth,
            h_enc_params = dict(
                depth = h_enc_depth,
                heads = h_enc_heads,
                mlp_dim = h_enc_mlp_dim,
                # dim_head = h_enc_dim_head
            ),
            l_enc_params = dict(
                depth = l_enc_depth,
                heads = l_enc_heads,
                mlp_dim = l_enc_mlp_dim,
                # dim_head = l_enc_dim_head
            ),
            dropout = dropout
        )
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        # self.to_cls_token = nn.Identity()

        # self.nn1 = nn.Linear(dim, num_classes)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
       

    def forward(self, x1, x2, mask = None):

        x1 = self.conv3d_features(x1)
        x1 = rearrange(x1, 'b c h w y ->b (c h) w y')
        x1 = self.conv2d_features(x1)
        x1 = rearrange(x1,'b c h w -> b (h w) c')

        x2 = self.conv2d_features2(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        
        wa1 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A1 = torch.einsum('bij,bjk->bik', x1, wa1)
        A1 = rearrange(A1, 'b h w -> b w h')  # Transpose
        A1 = A1.softmax(dim=-1)

        VV1 = torch.einsum('bij,bjk->bik', x1, self.token_wV)
        T1 = torch.einsum('bij,bjk->bik', A1, VV1)

        wa2 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A2 = torch.einsum('bij,bjk->bik', x2, wa2)
        A2 = rearrange(A2, 'b h w -> b w h')  # Transpose
        A2 = A2.softmax(dim=-1)

        VV2 = torch.einsum('bij,bjk->bik', x2, self.token_wV)
        T2 = torch.einsum('bij,bjk->bik', A2, VV2)

        cls_tokens1 = self.cls_token.expand(x1.shape[0], -1, -1)
        x1 = torch.cat((cls_tokens1, T1), dim=1)
        x1 += self.pos_embedding
        x1 = self.dropout(x1)
        # x1 = self.transformer(x1, mask)  # main game
        # x1 = self.to_cls_token(x1[:, 0])
        # x1 = self.nn1(x1)

        cls_tokens2 = self.cls_token.expand(x2.shape[0], -1, -1)
        x2 = torch.cat((cls_tokens2, T2), dim=1)
        x2 += self.pos_embedding
        x2 = self.dropout(x2)
        
        
        # x2 = self.transformer(x2, mask)  # main game
        # x2 = self.to_cls_token(x2[:, 0])
        # x2 = self.nn1(x2)
        x1, x2 = self.fusion_encoder(x1, x2)
        # x1 = self.transformer(x1, mask)
        # x2 = self.transformer(x2, mask)
        x1, x2 = map(lambda t: t[:, 0], (x1, x2))
        
        x = self.mlp_head(x1) + self.mlp_head(x2)
        # x = self.nn1(x)
        
        return x


if __name__ == '__main__':
    model = HCTnet()
    model.eval()
    print(model)
    input1 = torch.randn(64, 1, 30, 11, 11)
    input2 = torch.randn(64, 1, 11, 11)
    x = model(input1, input2)
    print(x.size())
    # summary(model, ((64, 1, 30, 11, 11), (64, 1, 11, 11)))

