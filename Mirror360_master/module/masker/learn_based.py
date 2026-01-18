import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module.masker.base_masker import BaseMaskerSystem, MaskGenerate 
# ========== 核心：Gumbel Top-K Masking ==========
def gumbel_topk_mask(logits, k, tau=0.1, hard=True):
    """
    可微分的Top-K选择
    Args:
        logits: (B, N) patch重要性分数
        k: 选择的patch数量
        tau: Gumbel-Softmax温度
        hard: 是否使用straight-through estimator
    Returns:
        mask: (B, N) 二值mask（forward时）或soft mask（backward时）
    """
    B, N = logits.shape
    
    # 1. 添加Gumbel噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    perturbed_logits = logits + gumbel_noise
    
    # 2. 获取top-k indices
    topk_values, topk_indices = torch.topk(perturbed_logits, k, dim=-1)
    
    # 3. 创建hard mask
    hard_mask = torch.zeros_like(logits)
    hard_mask.scatter_(1, topk_indices, 1.0)
    
    if hard:
        # 4. Straight-through estimator: forward用hard，backward用soft
        # soft mask使用sigmoid近似
        soft_mask = torch.sigmoid((logits.unsqueeze(1) - topk_values[:, -1:].unsqueeze(-1)) / tau)
        soft_mask = soft_mask.squeeze(1)
        
        # Straight-through
        mask = hard_mask.detach() + soft_mask - soft_mask.detach()
    else:
        mask = torch.sigmoid((logits.unsqueeze(1) - topk_values[:, -1:].unsqueeze(-1)) / tau).squeeze(1)
    #mask_size 
    return mask, topk_indices 


# ========== ViT组件 ==========
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, N, embed_dim)
        x = self.proj(x)  # (B, embed_dim, 14, 14)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ========== 主体：ViT Masker ==========
class ViTMasker(nn.Module):
    """
    基于ViT的自适应Patch选择器
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        use_cls_token=True
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        self.use_cls_token = use_cls_token
        
        # CLS token (可选)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Position embedding
        n_tokens = self.n_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, n_heads, mlp_ratio, qkv_bias,
                drop_rate, attn_drop_rate
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Score predictor: 预测每个patch的重要性
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def forward(self, x, k=None, tau=0.1, return_scores=False):
        """
        Args:
            x: (B, C, H, W) 输入图像
            k: 保留的patch数量（如果为None则不做mask）
            tau: Gumbel-Softmax温度
            return_scores: 是否返回原始分数
        Returns:
            mask: (B, N) 二值mask
            masked_tokens: (B, k, embed_dim) 选中的patch tokens
            topk_indices: (B, k) 选中的patch索引
        """
        B = x.shape[0]
        
        # 1. Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # 2. Add CLS token
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # 3. Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Transformer encoding
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 5. 分离CLS token和patch tokens
        if self.use_cls_token:
            cls_token = x[:, 0]
            patch_tokens = x[:, 1:]
        else:
            patch_tokens = x
            cls_token = None
        
        # 6. 预测每个patch的重要性分数
        patch_scores = self.score_head(patch_tokens).squeeze(-1)  # (B, N)
        
        # 7. Gumbel Top-K masking
        if k is not None and k < self.n_patches:
            mask, topk_indices = gumbel_topk_mask(patch_scores, k, tau)
            
        return mask ,patch_scores
           
class Learnimportancescore(nn.Module):
    def __init__(
        self, 
        img_size=640, 
        patch_size=16, 
        in_channels=3,
        embed_dim=12,
        depth=2,
        n_heads=4,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        use_cls_token=True
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        # CLS token (可选)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Position embedding
        n_tokens = self.n_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, n_heads, mlp_ratio, qkv_bias,
                drop_rate, attn_drop_rate
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Score predictor: 预测每个patch的重要性
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def scorer(self, x, k=None, return_scores=False):
        """
        Args:
            x: (B, C, H, W) 输入图像
            k: 保留的patch数量（如果为None则不做mask）
            tau: Gumbel-Softmax温度
            return_scores: 是否返回原始分数
        Returns:
            mask: (B, N) 二值mask
            masked_tokens: (B, k, embed_dim) 选中的patch tokens
            topk_indices: (B, k) 选中的patch索引
        """
        B,C,H,W = x.shape
        
        # 1. Patch embedding
        x = self.patch_embed(x)  # (B, N, embed_dim)
        
        # 2. Add CLS token
        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # 3. Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Transformer encoding
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 5. 分离CLS token和patch tokens
        if self.use_cls_token:
            cls_token = x[:, 0]
            patch_tokens = x[:, 1:]
        else:
            patch_tokens = x
            cls_token = None
        
        # 6. 预测每个patch的重要性分数
        patch_scores = self.score_head(patch_tokens).squeeze(-1)  # (B, N)
        
        patch_scores = rearrange(patch_scores,'b (h_p w_p) -> b h_p w_p',
                                 h_p=H//self.patch_size,
                                 w_p=W//self.patch_size)
        return patch_scores
    def forward(self, frames):
        B,T,C,H,W = frames.shape
        frames = rearrange(frames,'b t c h w -> (b t) c h w')
        patch_scores = self.scorer(frames)
        patch_scores = rearrange(patch_scores,'(b t) h_p w_p -> b t h_p w_p',
                                 b=B,
                                 t=T)

        return patch_scores
class learnDynamicMaskerSystem(BaseMaskerSystem):

    def __init__(self, patch_size=8, device='cuda', num_frames=10, 
                 gradient_strength=0.05, region_ratios=[0.1,0.2,0.3,0.4],num_strata=4):
        super().__init__()
        self.patch_size = patch_size
        self.device = device
        self.num_frames = num_frames
        self.num_strata = num_strata
        
        # 重要性评分器
        self.importance_scorer = Learnimportancescore(
            img_size=640, 
                patch_size=16, 
                in_channels=3,
                embed_dim=24,
                depth=2,
                n_heads=4,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                use_cls_token=True
        )
        #mask生成器
        self.region_masker = MaskGenerate(
            num_strata=4 if region_ratios is None else len(region_ratios),
            gradient_strength=gradient_strength,
            region_ratios=region_ratios  # 传入自定义比例
        )