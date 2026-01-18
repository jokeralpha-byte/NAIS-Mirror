import torch
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Tuple, Dict, Any, Optional, List
import cv2

# ============= 基础辅助函数 (无需修改) =============
def _pad_to_multiple(tensor: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad到指定倍数"""
    T, C, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, [0, pad_w, 0, pad_h])
    return tensor, (pad_h, pad_w)

def generate_raster_order(h: int, w: int) -> List[Tuple[int, int]]:
    """生成光栅扫描顺序（行优先）"""
    positions = []
    for y in range(h):
        for x in range(w):
            positions.append((y, x))
    return positions

def generate_diagonal_order(h: int, w: int) -> List[Tuple[int, int]]:
    """生成对角扫描顺序"""
    positions = []
    for d in range(h + w - 1):
        diag_points = []
        for y in range(max(0, d - w + 1), min(d + 1, h)):
            x = d - y
            diag_points.append((y, x))
        
        if d % 2 == 1:
            diag_points.reverse()
        
        positions.extend(diag_points)
    
    return positions

def compute_patch_features(patch: torch.Tensor, method: str = 'brightness') -> Any:
    """计算patch的特征用于排序"""
    if method == 'unrolled':
        return 0
    if patch.dim() == 1:
        C_dim = 3
        p = int(math.sqrt(patch.shape[0] / C_dim))
        patch = patch.reshape(p, p, C_dim)
    if method == 'brightness':
        return patch.mean().item()
    elif method == 'hue':
        r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
        max_c = torch.max(torch.stack([r, g, b]), dim=0)[0]
        min_c = torch.min(torch.stack([r, g, b]), dim=0)[0]
        delta = max_c - min_c
        hue = torch.zeros_like(max_c)
        mask = delta > 1e-7
        r_mask, g_mask, b_mask = (max_c == r) & mask, (max_c == g) & mask, (max_c == b) & mask
        hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
        hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
        hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
        return hue.mean().item()
    elif method == 'multi':
        brightness = patch.mean().item()
        saturation = patch.std().item()
        texture = (patch - patch.mean()).abs().mean().item()
        return (int(brightness * 10), int(saturation * 10), int(texture * 10))
    else:
        raise ValueError(f"Unknown method: {method}")

# ============= 🔴 已修改的 Squeeze 函数 =============
def squeeze_frame_independent(
    patch_size: int = 16,
    mask: Optional[torch.Tensor] = None,
    pic_tensor: Optional[torch.Tensor] = None,
    sort_method: str = 'brightness',
    fill_method: str = 'diagonal'
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if mask is None or pic_tensor is None:
        raise RuntimeError("请提供mask和pic_tensor")
    T, C, H, W = pic_tensor.shape
    p = patch_size
    
    padded, (pad_h, pad_w) = _pad_to_multiple(pic_tensor, p)
    h_p = (H + pad_h) // p
    w_p = (W + pad_w) // p
    
    patches = rearrange(padded, 't c (h p1) (w p2) -> t (h w) (p1 p2 c)', p1=p, p2=p)
    mask_flat = mask.reshape(T, h_p * w_p)
    
    kept_patches_list, original_indices_list, max_kept = [], [], 0
    for t in range(T):
        kept_mask = mask_flat[t].bool()
        kept_patches_t = patches[t][kept_mask]
        original_indices_t = torch.where(kept_mask)[0]
        kept_patches_list.append(kept_patches_t)
        original_indices_list.append(original_indices_t)
        max_kept = max(max_kept, kept_patches_t.shape[0])
    
    if max_kept == 0:
        raise ValueError("No patches to keep!")
    
    w_new = max(1, int(math.sqrt(max_kept)))
    h_new = (max_kept + w_new - 1) // w_new
    N_slots = h_new * w_new
    
    if fill_method == 'diagonal':
        fill_positions = generate_diagonal_order(h_new, w_new)
    else:
        fill_positions = generate_raster_order(h_new, w_new)
    
    # 准备一个空的、扁平化的画布
    squeezed_patches_flat = torch.zeros(T, N_slots, p * p * C, device=pic_tensor.device, dtype=pic_tensor.dtype)
    sort_indices_per_frame = []
    
    for t in range(T):
        kept_patches_t = kept_patches_list[t]
        original_indices_t = original_indices_list[t]
        N_kept_t = kept_patches_t.shape[0]
        
        if sort_method != 'unrolled':
            try:
                features = [compute_patch_features(patch, method=sort_method) for patch in kept_patches_t]
                sort_idx = sorted(range(N_kept_t), key=lambda i: features[i])
            except Exception as e:
                print(f"Warning: Feature computation failed for {sort_method}: {e}. Falling back to unrolled.")
                sort_idx = list(range(N_kept_t))
        else:
            sort_idx = list(range(N_kept_t))
        
        sorted_patches = kept_patches_t[sort_idx]
        sorted_original_indices = original_indices_t[sort_idx]
        
        # --- 核心修改 ---
        # 直接按照 fill_positions 的顺序，将排好序的 patch 放入扁平画布的指定位置
        for i in range(N_kept_t):
            y, x = fill_positions[i]
            flat_pos = y * w_new + x
            squeezed_patches_flat[t, flat_pos] = sorted_patches[i]
        # --- 修改结束 ---
        
        padded_indices = torch.full((max_kept,), -1, dtype=torch.long, device=pic_tensor.device)
        padded_indices[:N_kept_t] = sorted_original_indices
        sort_indices_per_frame.append(padded_indices)

    # 从填充好的扁平画布重组为图像，不再需要额外的重排步骤
    squeezed = rearrange(
        squeezed_patches_flat,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_new, w=w_new, p1=p, p2=p, c=C
    )

    # 保留的图片输出接口
    show_squeeze = squeezed * 255
    show_squeeze = rearrange(show_squeeze, 't c h w -> t h w c').to(torch.uint8).cpu().numpy()
    cv2.imwrite(f'/data2/mmvisitor/Jia_Daiang/MaStreaming/mastreaming_master/toshixiong/2{sort_method}_squeeze.png', cv2.cvtColor(show_squeeze[0], cv2.COLOR_RGB2BGR))
    
    sort_indices_stack = torch.stack(sort_indices_per_frame)
    metadata = {
        'method': f'frame_independent_{fill_method}_{sort_method}',
        'patch_size': p,
        'patched_shape': (h_p, w_p),
        'squeezed_shape': (h_new, w_new),
        'max_kept': max_kept,
        'sort_indices': sort_indices_stack,
        'fill_positions': fill_positions, # 恢复时需要此信息
        'region_shape': (H, W),
        'T': T,
    }
    return squeezed, metadata

# ============= 🔴 已修改的 Unsqueeze 函数 =============
def unsqueeze_frame_independent(
    mask: Optional[torch.Tensor] = None,
    squeezed_tensor: Optional[torch.Tensor] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    if metadata is None or squeezed_tensor is None:
        raise RuntimeError("需要squeezed_tensor 和 metadata")
    
    T, C, H_sq, W_sq = squeezed_tensor.shape
    p = metadata['patch_size']
    patched_shape = metadata['patched_shape']
    sort_indices = metadata['sort_indices']
    fill_positions = metadata['fill_positions']
    region_shape = metadata['region_shape']
    h_new, w_new = metadata['squeezed_shape']
    N_slots = h_new * w_new

    # 将输入的 squeezed tensor 变回 patch 网格
    squeezed_patches_grid = rearrange(
        squeezed_tensor,
        't c (h p1) (w p2) -> t h w (p1 p2 c)',
        p1=p, p2=p
    )
    
    # --- 核心修改 ---
    # 准备一个线性的容器
    linear_patches = torch.zeros(T, N_slots, p * p * C, device=squeezed_tensor.device, dtype=squeezed_tensor.dtype)
    # 按照 fill_positions 定义的扫描顺序，从网格中提取 patch，将其放回线性容器
    # 这样就还原了 squeeze 前的、排好序的 patch 列表
    for t in range(T):
        for i in range(N_slots):
            y, x = fill_positions[i]
            linear_patches[t, i] = squeezed_patches_grid[t, y, x]
    # --- 修改结束 ---

    h_p, w_p = patched_shape
    recon_patches = torch.zeros(T, h_p * w_p, p * p * C, device=squeezed_tensor.device, dtype=squeezed_tensor.dtype)
    
    for t in range(T):
        valid_mask = sort_indices[t] >= 0
        valid_original_indices = sort_indices[t][valid_mask]
        N_valid = valid_original_indices.shape[0]
        
        # 从还原的线性列表中取出有效的 patches
        valid_patches = linear_patches[t, :N_valid]
        
        recon_patches[t, valid_original_indices] = valid_patches
    
    recon = rearrange(
        recon_patches,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_p, w=w_p, p1=p, p2=p, c=C
    )
    
    return recon[:, :, :region_shape[0], :region_shape[1]]

