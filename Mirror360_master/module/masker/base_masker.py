import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np








class MaskGenerate(nn.Module):

    def __init__(self, num_strata=4, gradient_strength=0.15, region_ratios=None):
        """
        Args:
            num_strata: 分层数量（默认4层）
            gradient_strength: 梯度强度，控制层间mask比例差异（0.1-0.2合适）
            region_ratios: 每层占总patch数的比例，从低到高重要性
                          例如 [0.4, 0.3, 0.2, 0.1] 表示：
                          - 40%最不重要的patch为第0层
                          - 30%次不重要的patch为第1层
                          - 20%次重要的patch为第2层
                          - 10%最重要的patch为第3层
                          默认None时自动均分为 [0.25, 0.25, 0.25, 0.25]
        """
        super().__init__()
        self.num_strata = num_strata
        self.gradient_strength = gradient_strength
        
        # 设置区域比例
        if region_ratios is None:
            # 默认均分
            self.region_ratios = [1.0 / num_strata] * num_strata
        else:
            assert len(region_ratios) == num_strata, \
                f"region_ratios长度({len(region_ratios)})必须等于num_strata({num_strata})"
            assert abs(sum(region_ratios) - 1.0) < 1e-5, \
                f"region_ratios之和必须为1.0，当前为{sum(region_ratios)}"
            self.region_ratios = region_ratios
        
    def compute_layer_ratios(self, mask_ratio):
        
        step = self.gradient_strength
        
        # 计算基准值（最重要层的mask比例）
        base = mask_ratio - (self.num_strata - 1) * step / 2
        
        # 生成递减序列
        ratios = []

        for i in range(self.num_strata):
            # i=0是最不重要层（mask最多），i=N-1是最重要层（mask最少）
            ratio = base + (self.num_strata - 1 - i) * step
            # 限制范围 [0, 0.9]
            ratio = max(0.0, min(0.9, ratio))
            ratios.append(ratio)
        
        # 加权归一化，确保总mask数精确
        weighted_avg = sum(r * w for r, w in zip(ratios, self.region_ratios))
        if weighted_avg > 0:
            scale = mask_ratio / weighted_avg
            ratios = [r * scale for r in ratios]
        
        return ratios  # [高mask(不重要), ..., 低mask(重要)]
    
    def forward(self, scores, mask_ratio):#这个处理方式可以兼用逐帧mask和单帧mask
        """
        Args:
            scores: (B, T, H_p, W_p) 重要性分数，越高越重要
            mask_ratio: float, 总体mask比例
            
        Returns:
            mask: (B, T, H_p, W_p) 1=mask, 0=keep
        """
        B, T, H_p, W_p = scores.shape #scores: (B, T, H_p, W_p)
        N = H_p * W_p
        
        scores_flat = scores.reshape(B, T, -1)  # (B, T, N)
        
        #计算mask比例
        layer_ratios = self.compute_layer_ratios(mask_ratio)
        
        #计算每层数量
        layer_sizes = [max(1, int(N * ratio)) for ratio in self.region_ratios]
        # 确保总和等于N（处理舍入误差）
        diff = N - sum(layer_sizes)
        if diff != 0:
            # 将差值分配给最大的层
            max_idx = layer_sizes.index(max(layer_sizes))
            layer_sizes[max_idx] += diff
        
        #生成mask
        final_mask = torch.zeros_like(scores_flat)
        
        for b in range(B):
            for t in range(T):
                score_bt = scores_flat[b, t]  # (N,)
                
                # 按重要性排序（从高到低），保留了索引
                sorted_scores, sorted_indices = torch.sort(score_bt, descending=True)
                
                # 根据自定义比例划分层级
                layers = []
                start_idx = 0
                
                # 从最重要到最不重要分配
                for layer_idx in range(self.num_strata - 1, -1, -1):
                    layer_size = layer_sizes[layer_idx]
                    end_idx = start_idx + layer_size
                    
                    # 该层的patch索引（在原始顺序中）
                    layer_indices = sorted_indices[start_idx:end_idx]
                    layers.append((layer_idx, layer_indices))
                    
                    start_idx = end_idx
                
                # 在每层内随机mask指定比例
                for layer_idx, layer_indices in layers:
                    if len(layer_indices) == 0:
                        continue
                    
                    # 该层应该mask的数量
                    num_to_mask = int(len(layer_indices) * layer_ratios[layer_idx])
                    if num_to_mask == 0:
                        continue
                    
                    # 随机选择（保留探索红利）
                    perm = torch.randperm(len(layer_indices), device=scores.device)
                    #随机打乱后选择前mask索引个（既随机，又控制数量）
                    selected = layer_indices[perm[:num_to_mask]]

                    final_mask[b, t, selected] = 1.0
                    final_mask = final_mask.to(torch.bool)
        return final_mask.reshape(B, T, H_p, W_p)




class BaseMaskerSystem(nn.Module):
    def __init__(self, patch_size=8, device='cuda', num_frames=10, 
                 gradient_strength=0.15, region_ratios=[0.1,0.2,0.3,0.4]):  # 新增参数
        super().__init__()
        self.patch_size = patch_size
        self.device = device
        self.num_frames = num_frames
        
        # 重要性评分器
        self.importance_scorer = None
        self.region_masker = MaskGenerate(
            num_strata=4 if region_ratios is None else len(region_ratios),
            gradient_strength=gradient_strength,
            region_ratios=region_ratios  # 传入自定义比例
        )
    def forward(self, video, mask_ratio=0.5):
        """
        训练器接口：处理单帧
      
        Args:
            image: (B, C, H, W) 单帧图像，范围 [-1, 1] 或 [0, 1]
            mask_ratio: float, 要 mask 的比例
            temperature: float, 温度参数
            hard: bool, 是否硬选择
      
        Returns:
            mask_pixel: (B, C, H, W) 像素级 mask，1=mask, 0=keep
            importance_scores: (B, 1, H_p, W_p) 重要性分数
        """

      
        # 归一化到[0, 1]
        if video.min() < 0:
            video = (video + 1) / 2
        B,T, C, H, W = video.shape
        # 1. 计算重要性分数
        importance_scores = self.importance_scorer(video)  # (B, T, H/8, W/8)
        mask = self.region_masker(importance_scores, mask_ratio) # (B, T, H/8, W/8)
        return mask
    def reshape_mask(self,mask):
        mask_img = mask.unsqueeze(4).expand(-1,-1,-1,-1,self.patch_size*self.patch_size)  # (B,T, H_p, W_p,1)
        mask_img = rearrange(mask_img, 'b t h w (p1 p2) -> b t (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)  # (B, 1, H, W)
        return mask_img