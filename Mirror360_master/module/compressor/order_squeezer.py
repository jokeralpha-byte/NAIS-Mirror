# ============= 修复后的完整代码 =============

import torch
import torch.nn.functional as F
from einops import rearrange
import math

def _pad_to_multiple(tensor, multiple):
    """Pad到指定倍数"""
    T, C, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, [0, pad_w, 0, pad_h])
    return tensor, (pad_h, pad_w)


def _unify_kept_patches(kept_patches, pad_value=0.0):
    """
    统一不同帧的kept patches数量
    
    Args:
        kept_patches: List[Tensor], 每个shape为 (N_keep_i, p*p*C)
        pad_value: padding的值
        
    Returns:
        unified: (T, N_keep_max, p*p*C)
        N_keeps: (T,) 每帧实际保留的数量
    """
    T = len(kept_patches)
    N_keeps = torch.tensor([kp.shape[0] for kp in kept_patches])
    N_keep_max = N_keeps.max().item()
    
    device = kept_patches[0].device
    dtype = kept_patches[0].dtype
    feature_dim = kept_patches[0].shape[1]
    
    # 创建统一的tensor
    unified = torch.full(
        (T, N_keep_max, feature_dim),
        pad_value,
        device=device,
        dtype=dtype
    )
    
    # 填充每帧的kept patches
    for t in range(T):
        n_keep = N_keeps[t].item()
        unified[t, :n_keep] = kept_patches[t]
    
    return unified, N_keeps


# ============= 方法1: 空间保序（修复版）=============

def squeeze_spatial_order(
    patch_size=16,
    mask=None,
    pic_tensor=None
):
    """空间保序的squeeze - 修复版"""
    if mask is None:
        raise RuntimeError("请生成mask之后再squeeze")

    T, C, H, W = pic_tensor.shape
    p = patch_size

    # 1. Pad
    padded, (pad_h, pad_w) = _pad_to_multiple(pic_tensor, p)
    h_p = (H + pad_h) // p
    w_p = (W + pad_w) // p
    patched_shape = (h_p, w_p)

    # 2. 转为patch序列
    patches = rearrange(
        padded,
        't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
        p1=p, p2=p, h=h_p, w=w_p
    )

    mask_flat = mask.reshape(T, h_p * w_p)

    # 3. 按空间顺序提取patches
    kept_patches = []
    original_indices = []
    
    for t in range(T):
        indices = torch.where(mask_flat[t])[0]
        original_indices.append(indices)
        kept_patches.append(patches[t][indices])
    
    # 🔥 统一kept patches数量
    kept, N_keeps = _unify_kept_patches(kept_patches)  # (T, N_keep_max, p*p*C)
    N_keep = N_keeps.max().item()

    # 4. 计算紧凑矩形
    w_new = max(1, int(math.sqrt(N_keep)))
    h_new = (N_keep + w_new - 1) // w_new
    need_pad = h_new * w_new - N_keep

    # 5. Padding到矩形
    if need_pad > 0:
        pad = torch.zeros(T, need_pad, kept.shape[-1],
                        device=pic_tensor.device, dtype=pic_tensor.dtype)
        kept = torch.cat([kept, pad], dim=1)

    # 6. 重组
    squeezed = rearrange(
        kept,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_new, w=w_new, p1=p, p2=p, c=C
    )

    # 7. 元数据
    metadata = {
        'method': 'spatial_order',
        'patched_shape': patched_shape,
        'N_keep': N_keep,
        'N_keeps': N_keeps,  # 🔥 每帧实际的数量
        'original_indices': original_indices,  # List[Tensor]
        'region_shape': (H, W),
        'squeezed_shape': (h_new, w_new),
    }

    return squeezed, metadata


def unsqueeze_spatial_order(
    mask=None,
    squeezed_tensor=None,
    metadata=None
):
    """空间保序的unsqueeze - 修复版"""
    if mask is None or metadata is None:
        raise RuntimeError("需要mask和metadata")

    T, C, H_sq, W_sq = squeezed_tensor.shape
    p = 16
    
    patched_shape = metadata['patched_shape']
    N_keeps = metadata['N_keeps']  # (T,)
    original_indices = metadata['original_indices']  # List[Tensor]
    region_shape = metadata['region_shape']

    # 1. 提取patches
    patches = rearrange(
        squeezed_tensor,
        't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
        p1=p, p2=p
    )

    # 2. 还原到原始mask位置
    h_p, w_p = patched_shape
    full = torch.zeros(T, h_p * w_p, p * p * C,
                      device=squeezed_tensor.device, 
                      dtype=squeezed_tensor.dtype)
    
    mask_flat = mask.reshape(T, h_p * w_p)
    
    # 🔥 每帧单独处理，使用正确的N_keep
    for t in range(T):
        n_keep = N_keeps[t].item()
        valid_patches = patches[t, :n_keep]  # 只取有效的patches
        full[t, original_indices[t]] = valid_patches

    # 3. 重组
    recon = rearrange(
        full,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_p, w=w_p, p1=p, p2=p, c=C
    )

    return recon[:, :, :region_shape[0], :region_shape[1]]


# ============= 方法2: 相似度排序（修复版）=============

def squeeze_similarity_order(
    patch_size=16,
    mask=None,
    pic_tensor=None,
    sort_by='brightness'
):
    """相似度排序的squeeze - 修复版"""
    if mask is None:
        raise RuntimeError("请生成mask之后再squeeze")

    T, C, H, W = pic_tensor.shape
    p = patch_size

    # 1-2. Pad和分patch
    padded, (pad_h, pad_w) = _pad_to_multiple(pic_tensor, p)
    h_p = (H + pad_h) // p
    w_p = (W + pad_w) // p
    patched_shape = (h_p, w_p)

    patches = rearrange(
        padded,
        't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
        p1=p, p2=p, h=h_p, w=w_p
    )

    mask_flat = mask.reshape(T, h_p * w_p)

    # 3. 提取kept patches
    kept_patches = []
    original_indices = []
    
    for t in range(T):
        indices = torch.where(mask_flat[t])[0]
        original_indices.append(indices)
        kept_patches.append(patches[t][indices])
    
    # 🔥 统一数量
    kept, N_keeps = _unify_kept_patches(kept_patches)
    N_keep = N_keeps.max().item()

    # 4. 计算相似度并排序
    kept_reshaped = kept.reshape(T, N_keep, p*p, C)
    
    if sort_by == 'brightness':
        sort_keys = kept_reshaped.mean(dim=(2, 3))
    elif sort_by == 'color':
        color_mean = kept_reshaped.mean(dim=2)
        sort_keys = (color_mean[..., 0] * 1000 + 
                    color_mean[..., 1] * 100 + 
                    color_mean[..., 2])
    elif sort_by == 'hybrid':
        brightness = kept_reshaped.mean(dim=(2, 3))
        color_mean = kept_reshaped.mean(dim=2)
        sort_keys = (brightness * 10000 + 
                    color_mean[..., 0] * 100 + 
                    color_mean[..., 1] * 10 + 
                    color_mean[..., 2])
    
    # 排序
    sorted_kept = []
    sort_orders = []
    
    for t in range(T):
        n_keep = N_keeps[t].item()
        # 🔥 只排序有效的patches
        valid_keys = sort_keys[t, :n_keep]
        sort_order = torch.argsort(valid_keys)
        
        # 对有效patches排序
        sorted_valid = kept[t, :n_keep][sort_order]
        
        # padding部分保持不变
        sorted_frame = kept[t].clone()
        sorted_frame[:n_keep] = sorted_valid
        
        sorted_kept.append(sorted_frame)
        
        # 保存完整的sort_order（包含padding索引）
        full_sort_order = torch.arange(N_keep, device=kept.device)
        full_sort_order[:n_keep] = sort_order
        sort_orders.append(full_sort_order)
    
    kept = torch.stack(sorted_kept, dim=0)
    sort_orders = torch.stack(sort_orders)

    # 5-6. 重组
    w_new = max(1, int(math.sqrt(N_keep)))
    h_new = (N_keep + w_new - 1) // w_new
    need_pad = h_new * w_new - N_keep

    if need_pad > 0:
        pad = torch.zeros(T, need_pad, kept.shape[-1],
                        device=pic_tensor.device, dtype=pic_tensor.dtype)
        kept = torch.cat([kept, pad], dim=1)

    squeezed = rearrange(
        kept,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_new, w=w_new, p1=p, p2=p, c=C
    )

    # 7. 元数据
    metadata = {
        'method': 'similarity_order',
        'patched_shape': patched_shape,
        'N_keep': N_keep,
        'N_keeps': N_keeps,
        'original_indices': original_indices,
        'sort_orders': sort_orders,
        'region_shape': (H, W),
        'squeezed_shape': (h_new, w_new),
    }

    return squeezed, metadata


def unsqueeze_similarity_order(
    mask=None,
    squeezed_tensor=None,
    metadata=None
):
    """相似度排序的unsqueeze - 修复版"""
    if mask is None or metadata is None:
        raise RuntimeError("需要mask和metadata")

    T, C, H_sq, W_sq = squeezed_tensor.shape
    p = 16
    
    patched_shape = metadata['patched_shape']
    N_keeps = metadata['N_keeps']
    original_indices = metadata['original_indices']
    sort_orders = metadata['sort_orders']
    region_shape = metadata['region_shape']

    # 1. 提取patches
    patches = rearrange(
        squeezed_tensor,
        't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
        p1=p, p2=p
    )

    # 2. 逆排序
    unsorted_patches = []
    for t in range(T):
        n_keep = N_keeps[t].item()
        
        # 只处理有效的patches
        valid_patches = patches[t, :n_keep]
        valid_sort_order = sort_orders[t, :n_keep]
        
        # 逆排序
        inverse_order = torch.argsort(valid_sort_order)
        unsorted_valid = valid_patches[inverse_order]
        
        unsorted_patches.append(unsorted_valid)

    # 3. 还原到mask位置
    h_p, w_p = patched_shape
    full = torch.zeros(T, h_p * w_p, p * p * C,
                      device=squeezed_tensor.device, 
                      dtype=squeezed_tensor.dtype)
    
    mask_flat = mask.reshape(T, h_p * w_p)
    
    for t in range(T):
        full[t, original_indices[t]] = unsorted_patches[t]

    # 4. 重组
    recon = rearrange(
        full,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_p, w=w_p, p1=p, p2=p, c=C
    )

    return recon[:, :, :region_shape[0], :region_shape[1]]


# ============= 方法3: Hilbert曲线（修复版）=============

def hilbert_index(x, y, order=4):
    """计算Hilbert曲线索引"""
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        return x, y
    
    n = 2 ** order
    rx, ry, d = 0, 0, 0
    s = n // 2
    
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s //= 2
    
    return d


def squeeze_hilbert_order(
    patch_size=16,
    mask=None,
    pic_tensor=None
):
    """Hilbert曲线排序 - 修复版"""
    if mask is None:
        raise RuntimeError("请生成mask之后再squeeze")

    T, C, H, W = pic_tensor.shape
    p = patch_size

    padded, (pad_h, pad_w) = _pad_to_multiple(pic_tensor, p)
    h_p = (H + pad_h) // p
    w_p = (W + pad_w) // p
    patched_shape = (h_p, w_p)

    patches = rearrange(
        padded,
        't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
        p1=p, p2=p, h=h_p, w=w_p
    )

    mask_flat = mask.reshape(T, h_p * w_p)

    # 预计算Hilbert索引
    order = max(3, int(math.ceil(math.log2(max(h_p, w_p)))))
    
    all_hilbert_indices = torch.zeros(h_p * w_p, 
                                      dtype=torch.long, 
                                      device=pic_tensor.device)
    
    for idx in range(h_p * w_p):
        i = idx // w_p
        j = idx % w_p
        all_hilbert_indices[idx] = hilbert_index(j, i, order)

    # 提取并排序
    kept_patches = []
    original_indices = []
    hilbert_sort_orders = []
    
    for t in range(T):
        indices = torch.where(mask_flat[t])[0]
        original_indices.append(indices)
        
        hilbert_vals = all_hilbert_indices[indices]
        sort_order = torch.argsort(hilbert_vals)
        hilbert_sort_orders.append(sort_order)
        
        kept_patches.append(patches[t][indices][sort_order])
    
    # 统一数量
    kept, N_keeps = _unify_kept_patches(kept_patches)
    N_keep = N_keeps.max().item()

    # 重组
    w_new = max(1, int(math.sqrt(N_keep)))
    h_new = (N_keep + w_new - 1) // w_new
    need_pad = h_new * w_new - N_keep

    if need_pad > 0:
        pad = torch.zeros(T, need_pad, kept.shape[-1],
                        device=pic_tensor.device, dtype=pic_tensor.dtype)
        kept = torch.cat([kept, pad], dim=1)

    squeezed = rearrange(
        kept,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_new, w=w_new, p1=p, p2=p, c=C
    )

    metadata = {
        'method': 'hilbert_order',
        'patched_shape': patched_shape,
        'N_keep': N_keep,
        'N_keeps': N_keeps,
        'original_indices': original_indices,
        'hilbert_sort_orders': hilbert_sort_orders,
        'region_shape': (H, W),
        'squeezed_shape': (h_new, w_new),
    }

    return squeezed, metadata


def unsqueeze_hilbert_order(
    mask=None,
    squeezed_tensor=None,
    metadata=None
):
    """Hilbert曲线unsqueeze - 修复版"""
    if mask is None or metadata is None:
        raise RuntimeError("需要mask和metadata")

    T, C, H_sq, W_sq = squeezed_tensor.shape
    p = 16
    
    patched_shape = metadata['patched_shape']
    N_keeps = metadata['N_keeps']
    original_indices = metadata['original_indices']
    hilbert_sort_orders = metadata['hilbert_sort_orders']
    region_shape = metadata['region_shape']

    # 1. 提取patches
    patches = rearrange(
        squeezed_tensor,
        't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
        p1=p, p2=p
    )

    # 2. 逆Hilbert排序
    unsorted_patches = []
    for t in range(T):
        n_keep = N_keeps[t].item()
        valid_patches = patches[t, :n_keep]
        inverse_order = torch.argsort(hilbert_sort_orders[t])
        unsorted_patches.append(valid_patches[inverse_order])

    # 3. 还原
    h_p, w_p = patched_shape
    full = torch.zeros(T, h_p * w_p, p * p * C,
                      device=squeezed_tensor.device, 
                      dtype=squeezed_tensor.dtype)
    
    mask_flat = mask.reshape(T, h_p * w_p)
    
    for t in range(T):
        full[t, original_indices[t]] = unsorted_patches[t]

    # 4. 重组
    recon = rearrange(
        full,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=h_p, w=w_p, p1=p, p2=p, c=C
    )

    return recon[:, :, :region_shape[0], :region_shape[1]]


# ============= 边界平滑 =============

def smooth_patch_boundaries(squeezed, patch_size=16, blend_width=1):
    """边界平滑"""
    if blend_width == 0:
        return squeezed
    
    T, C, H, W = squeezed.shape
    p = patch_size
    b = blend_width
    
    smoothed = squeezed.clone()
    
    # 水平边界
    for i in range(p, W, p):
        if i >= W:
            continue
        left_start = max(0, i - b)
        left_end = i
        right_start = i
        right_end = min(W, i + b)
        
        if left_start < left_end and right_start < right_end:
            width = min(b, left_end - left_start, right_end - right_start)
            left = smoothed[:, :, :, left_end-width:left_end]
            right = smoothed[:, :, :, right_start:right_start+width]
            
            alpha = torch.linspace(0, 1, width, device=squeezed.device)
            alpha = alpha.view(1, 1, 1, -1)
            blended = left * (1 - alpha) + right * alpha
            smoothed[:, :, :, left_end-width:left_end] = blended
    
    # 垂直边界
    for i in range(p, H, p):
        if i >= H:
            continue
        top_start = max(0, i - b)
        top_end = i
        bottom_start = i
        bottom_end = min(H, i + b)
        
        if top_start < top_end and bottom_start < bottom_end:
            height = min(b, top_end - top_start, bottom_end - bottom_start)
            top = smoothed[:, :, top_end-height:top_end, :]
            bottom = smoothed[:, :, bottom_start:bottom_start+height, :]
            
            alpha = torch.linspace(0, 1, height, device=squeezed.device)
            alpha = alpha.view(1, 1, -1, 1)
            blended = top * (1 - alpha) + bottom * alpha
            smoothed[:, :, top_end-height:top_end, :] = blended
    
    return smoothed


# ============= 统一接口 =============

def squeeze_unified(
    patch_size=16,
    mask=None,
    pic_tensor=None,
    method='spatial',
    smooth=False,
    smooth_width=1,
    similarity_mode='hybrid'
):
    """统一的squeeze接口"""
    if method == 'spatial':
        squeezed, metadata = squeeze_spatial_order(patch_size, mask, pic_tensor)
    elif method == 'similarity':
        squeezed, metadata = squeeze_similarity_order(
            patch_size, mask, pic_tensor, sort_by=similarity_mode
        )
    elif method == 'hilbert':
        squeezed, metadata = squeeze_hilbert_order(patch_size, mask, pic_tensor)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if smooth:
        squeezed = smooth_patch_boundaries(squeezed, patch_size, smooth_width)
        metadata['smoothed'] = True
        metadata['smooth_width'] = smooth_width
    else:
        metadata['smoothed'] = False
    
    return squeezed, metadata


def unsqueeze_unified(
    mask=None,
    squeezed_tensor=None,
    metadata=None
):
    """统一的unsqueeze接口"""
    if metadata is None:
        raise RuntimeError("需要metadata")
    
    method = metadata['method']
    
    if method == 'spatial_order':
        return unsqueeze_spatial_order(mask, squeezed_tensor, metadata)
    elif method == 'similarity_order':
        return unsqueeze_similarity_order(mask, squeezed_tensor, metadata)
    elif method == 'hilbert_order':
        return unsqueeze_hilbert_order(mask, squeezed_tensor, metadata)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============= 测试 =============

def debug_full_pipeline():
    """调试完整流程"""
    print("="*160)
    print("🧪 调试Squeeze/Unsqueeze流程")
    print("="*160)
    
    # 1. 创建输入
    T, C, H, W = 10, 3, 640, 640
    pic_tensor = torch.rand(T, C, H, W)
    print(f"📥 输入shape: {pic_tensor.shape}")
    
    # 2. 生成mask（随机，所以每帧保留数量可能不同）
    patch_size = 16
    h_p = (H + 7) // 16
    w_p = (W + 7) // 16
    
    mask = torch.rand(T, h_p, w_p) > 0.5
    print(f"🎭 Mask shape: {mask.shape}")
    
    # 统计每帧保留的数量
    for t in range(min(3, T)):
        n = mask[t].sum().item()
        print(f"   帧{t}: {n}/{h_p*w_p} patches ({n/(h_p*w_p)*100:.1f}%)")
    
    # 3. 测试所有方法
    methods = ['spatial', 'similarity', 'hilbert']
    
    for method in methods:
        print(f"\n{'='*160}")
        print(f"🔧 测试方法: {method}")
        print(f"{'='*160}")
        
        try:
            # Squeeze
            squeezed, metadata = squeeze_unified(
                patch_size=patch_size,
                mask=mask,
                pic_tensor=pic_tensor,
                method=method,
                smooth=True
            )
            print(f"✅ Squeeze成功: {squeezed.shape}")
            print(f"   N_keeps: {metadata['N_keeps'].tolist()[:3]}... (前3帧)")
            print(f"   N_keep_max: {metadata['N_keep']}")
            
            # Unsqueeze
            reconstructed = unsqueeze_unified(
                mask=mask,
                squeezed_tensor=squeezed,
                metadata=metadata
            )
            print(f"✅ Unsqueeze成功: {reconstructed.shape}")
            
            # 验证
            if reconstructed.shape == pic_tensor.shape:
                print(f"✅ 形状匹配！")
            else:
                print(f"❌ 形状不匹配！")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    debug_full_pipeline()
