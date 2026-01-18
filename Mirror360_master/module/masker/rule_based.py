import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module.masker.base_masker import BaseMaskerSystem, MaskGenerate



class RuleBasedImportanceScorer(nn.Module):
    
    """
    纯规则驱动的 patch 重要性评分器
    输入：视频 clip (B, T, C, H, W)
    输出：每个 patch 的重要性分数 (B, T, H_p, W_p)
    """
    def __init__(self, patch_size=16, num_frames=5,
                 w_edge=0.4, w_color=0.3, w_entropy=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.w_edge = w_edge
        self.w_color = w_color
        self.w_entropy = w_entropy

        # Sobel 算子 (3x3)
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                     [ 0,  0,  0],
                                     [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

    @torch.no_grad()
    def forward(self, video):
        """
        Args:
            video: (B, T, C, H, W) float32, [0,1] or [0,255]
        Returns:
            scores: (B, T, H//p, W//p) in [0,1]
        """
        B, T, C, H, W = video.shape
        p = self.patch_size
        device = video.device

        # 1. 归一化到 [0,1]
        if video.max() > 1.0:
            video = video / 255.0

        # 2. 转为灰度（重要性对亮度敏感）
        gray = 0.299 * video[:, :, 0] + 0.587 * video[:, :, 1] + 0.114 * video[:, :, 2]  # (B,T,H,W)

        # 3. 计算三项规则分数
        edge_score = self._compute_edge_score(gray)           # (B,T,H,W)
        color_disc_score = self._compute_color_discontinuity(video)  # (B,T,H,W)
        entropy_score = self._compute_entropy_score(gray)     # (B,T,H,W)

        # 4. 加权融合
        importance = (self.w_edge * edge_score +
                      self.w_color * color_disc_score +
                      self.w_entropy * entropy_score)

        # 5. 下采样到 patch 级别 (mean pool)
        importance_padded = F.pad(importance, 
                                  (0, (p - W % p) % p, 0, (p - H % p) % p),
                                  mode='reflect')
        B, T, H_pad, W_pad = importance_padded.shape
        importance_patches = importance_padded.view(B, T, H_pad//p, p, W_pad//p, p)
        patch_scores = importance_patches.mean(dim=(3, 5))  # (B,T,h_p,w_p)
        #对所有位置取重要性分数的严格平均
        patch_scores = patch_scores.mean(dim=1).unsqueeze(1)  # (B,1,h_p,w_p)
        patch_scores = patch_scores.expand(-1,T,-1,-1)
        return patch_scores.clamp(0, 1)

    def _compute_edge_score(self, gray):
        """结构复杂度：Sobel 边缘强度"""
        self.sobel_x = self.sobel_x.to(gray.device)
        self.sobel_y = self.sobel_y.to(gray.device)

        # (B*T, 1, H, W)
        gray_flat = gray.view(-1, 1, gray.shape[2], gray.shape[3])
        pad = (1, 1, 1, 1)
        gray_pad = F.pad(gray_flat, pad, mode='reflect')

        gx = F.conv2d(gray_pad, self.sobel_x, padding=0)
        gy = F.conv2d(gray_pad, self.sobel_y, padding=0)
        edge = torch.sqrt(gx**2 + gy**2 + 1e-8)
        edge = edge.view(gray.shape[0], gray.shape[1], *edge.shape[2:])  # (B,T,H,W)

        # 归一化到 [0,1]
        edge_max = edge.view(edge.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
        edge = edge / (edge_max + 1e-8)
        return edge.clamp(0, 1)

    def _compute_color_discontinuity(self, video):
        """颜色不连续性：相邻像素差"""
        r = video[:, :, 0]
        g = video[:, :, 1]
        b = video[:, :, 2]

        # 水平差 + 垂直差
        diff_h = torch.abs(r[..., :, 1:] - r[..., :, :-1]) + \
                 torch.abs(g[..., :, 1:] - g[..., :, :-1]) + \
                 torch.abs(b[..., :, 1:] - b[..., :, :-1])
        diff_v = torch.abs(r[..., 1:, :] - r[..., :-1, :]) + \
                 torch.abs(g[..., 1:, :] - g[..., :-1, :]) + \
                 torch.abs(b[..., 1:, :] - b[..., :-1, :])

        # 补齐尺寸
        diff_h = F.pad(diff_h, (0, 1, 0, 0), mode='constant', value=0)
        diff_v = F.pad(diff_v, (0, 0, 0, 1), mode='constant', value=0)

        disc = diff_h + diff_v  # (B,T,H,W)
        disc = disc / (disc.view(disc.shape[0], -1).max(dim=1)[0].view(-1,1,1,1) + 1e-8)
        return disc.clamp(0, 1)

    def _compute_entropy_score(self, gray):
        """Softmax 熵：越均匀越不重要 → 反转后越高越重要"""
        B, T, H, W = gray.shape
        p = self.patch_size

        # 先 pad 再分 patch
        gray_pad = F.pad(gray, (0, (p - W % p) % p, 0, (p - H % p) % p), mode='reflect')
        H_pad, W_pad = gray_pad.shape[2], gray_pad.shape[3]
        patches = gray_pad.view(B, T, H_pad//p, p, W_pad//p, p)  # (B,T,h,p,w,p)

        # 展平每个 patch 为 histogram
        patches_flat = patches.reshape(B*T* (H_pad//p) * (W_pad//p), p*p)
        patches_flat = patches_flat + 1e-8  # 避免零

        # softmax 模拟概率分布
        probs = F.softmax(patches_flat * 10, dim=-1)  # 温度 0.1

        # 熵
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # (N_patches,)
        entropy = entropy / math.log(p*p)  # 归一化到 [0,1]

        # 熵越高 → 越均匀 → 越不重要 → 重要性 = 1 - entropy
        importance = 1.0 - entropy

        # 还原到 (B,T,h,w)
        importance = importance.view(B, T, H_pad//p, W_pad//p)
        # 上采样回原图尺寸（复制）
        importance = importance.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        importance = importance[:, :, :H, :W]

        return importance

class RuleBasedDynamicImportanceScorer(RuleBasedImportanceScorer):
    """
    基于规则的动态重要性评分器
    输入：视频 clip (B, T, C, H, W)
    输出：每个 patch 的重要性分数 (B, T, H_p, W_p)
    """
    @torch.no_grad()
    def forward(self, video):
        """
        Args:
            video: (B, T, C, H, W) float32, [0,1] or [0,255]
        Returns:
            scores: (B, T, H//p, W//p) in [0,1]
        """
        B, T, C, H, W = video.shape
        p = self.patch_size
        device = video.device

        # 1. 归一化到 [0,1]
        if video.max() > 1.0:
            video = video / 255.0

        # 2. 转为灰度（重要性对亮度敏感）
        gray = 0.299 * video[:, :, 0] + 0.587 * video[:, :, 1] + 0.114 * video[:, :, 2]  # (B,T,H,W)

        # 3. 计算三项规则分数
        edge_score = self._compute_edge_score(gray)           # (B,T,H,W)
        color_disc_score = self._compute_color_discontinuity(video)  # (B,T,H,W)
        entropy_score = self._compute_entropy_score(gray)     # (B,T,H,W)

        # 4. 加权融合
        importance = (self.w_edge * edge_score +
                      self.w_color * color_disc_score +
                      self.w_entropy * entropy_score)

        # 5. 下采样到 patch 级别 (mean pool)
        importance_padded = F.pad(importance, 
                                  (0, (p - W % p) % p, 0, (p - H % p) % p),
                                  mode='reflect')
        B, T, H_pad, W_pad = importance_padded.shape
        importance_patches = importance_padded.view(B, T, H_pad//p, p, W_pad//p, p)
        patch_scores = importance_patches.mean(dim=(3, 5))  # (B,T,h_p,w_p)
        return patch_scores
    
class RuleMaskerSystem(BaseMaskerSystem):
    def __init__(self, patch_size=8, device='cuda', num_frames=10, 
                 gradient_strength=0.05, region_ratios=[0.1,0.2,0.3,0.4]):
        super().__init__()
        self.patch_size = patch_size
        self.device = device
        self.num_frames = num_frames
        
        # 重要性评分器
        self.importance_scorer = RuleBasedImportanceScorer(
            patch_size=patch_size,
            num_frames=num_frames
        )
        #mask生成器
        self.region_masker = MaskGenerate(
            num_strata=4 if region_ratios is None else len(region_ratios),
            gradient_strength=gradient_strength,
            region_ratios=region_ratios  # 传入自定义比例
        )
class RuleDynamicMaskerSystem(BaseMaskerSystem):
    def __init__(self, patch_size=16, device='cuda', num_frames=10, 
                 gradient_strength=0.05, region_ratios=[0.1,0.2,0.3,0.4],num_strata=4,):
        super().__init__()
        self.patch_size = patch_size
        self.device = device
        self.num_frames = num_frames
        self.num_strata = num_strata
        # 重要性评分器
        self.importance_scorer = RuleBasedDynamicImportanceScorer(
            patch_size=patch_size,
            num_frames=num_frames
        )
        #mask生成器
        self.region_masker = MaskGenerate(
            num_strata=4 if region_ratios is None else len(region_ratios),
            gradient_strength=gradient_strength,
            region_ratios=region_ratios  # 传入自定义比例
        )