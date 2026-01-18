from einops import rearrange
import torch
import math
import torch.nn.functional as F

#对剩下的patch进行padd后输出最适合压缩的size
def _pad_to_multiple(tensor, multiple):
    T, C, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, [0, pad_w, 0, pad_h])
    return tensor, (pad_h, pad_w)
    
#进行挤压
def squeeze(patch_size = 8,
            mask = None,
            pic_tensor = None):

        if mask is None:
            raise RuntimeError("请生成mask之后再squeeze")

        T, C, H, W = pic_tensor.shape
        p = patch_size

        # 1. pad 到 patch 倍数
        padded, (pad_h, pad_w) = _pad_to_multiple(pic_tensor, p)
        patched_shape = (((H + pad_h) // p) , ((W + pad_w) // p))
        h_p_pad = (H + pad_h + p - 1) // p
        w_p_pad = (W + pad_w + p - 1) // p
        
        # 2. 转为 patch 序列
        patches = rearrange(
            padded,
            't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
            p1=p, p2=p, h=h_p_pad, w=w_p_pad
        )  # (T, N_pad, p*p*C)
        mask = mask.reshape(T, h_p_pad * w_p_pad)  # (T, N_pad)
        # 3. 保留 patch
        kept = patches[mask]  # (T, N_keep, p*p*C)
        kept = kept.reshape(T,-1,patch_size*patch_size*C)
        # 4. 计算紧凑矩形尺寸
        N_keep = kept.shape[1]
        if N_keep == 0:
            raise ValueError("No patches to keep!")
        w_new = max(1, int(math.sqrt(N_keep)))
        h_new = (N_keep + w_new - 1) // w_new  #聪明的向上取整方法
        need_pad = h_new * w_new - N_keep

        # 5. padding 到完整矩形
        if need_pad > 0:
            pad = torch.zeros(T, need_pad, kept.shape[-1],
                            device=pic_tensor.device, dtype=pic_tensor.dtype)
            kept = torch.cat([kept, pad], dim=1)

        # 6. 重组成图像
        squeezed = rearrange(
            kept,
            't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
            h=h_new, w=w_new, p1=p, p2=p, c=C
        )

        return squeezed, patched_shape, N_keep

def unsqueeze(mask=None,
            squeezed_tensor = None,
            N_keep=None,
            patch_size=8,
            patched_shape=None,
            region_shape = None
            ):
    """
    恢复原图（与 StaticRandomMasker 完全一致）
    """
    if mask is None:
        raise RuntimeError("Mask not available")

    T, C, H, W = squeezed_tensor.shape
    p = patch_size

    patches = rearrange(
        squeezed_tensor,
        't c (h p1) (w p2) -> t (h w) (p1 p2 c)',
        p1=p, p2=p
    )  # (T, N_squeezed, p*p*C)

    # 2. 取前 N_keep 个 patch
    
    valid = patches[:, :N_keep]

    # 3. 还原到完整网格
    full = torch.zeros(T, patched_shape[0]*patched_shape[1], p*p*C,
                        device=squeezed_tensor.device, dtype=squeezed_tensor.dtype)
    mask = mask.reshape(T,patched_shape[0]*patched_shape[1])
    valid = valid.reshape(valid.shape[1]*valid.shape[0],patch_size*patch_size*C)
    full[mask] = valid

    # 4. 重组成原图
    recon = rearrange(
        full,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=patched_shape[0], w=patched_shape[1], p1=p, p2=p, c=C
    )

    return recon[:, :, :region_shape[0], :region_shape[1]]
    