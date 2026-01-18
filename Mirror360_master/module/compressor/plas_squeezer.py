import numpy as np
import torch
import time
import kornia
import functools
import itertools
from tqdm import tqdm
import os
from collections import defaultdict

# TODO settle on either torchvision or kornia

SHOW_VIS = False
WRITE_IMG_TMP = False
WRITE_FOLDER = "/tmp/plas-dbg"
WRITE_IDX = defaultdict(int)

if SHOW_VIS or WRITE_IMG_TMP:
    import cv2


# TODO perf: distance calculation done for the whole array after sorting for early breaking
#            we do the distance calculation already for the sorting, so we could use that

# TODO perf don't drag target and params around, only map indices?

# TODO perf remember the whole params_to_blocky remapping, redo it later?
# TODO could then also map from 0 to 1 to 2 directly, without going back to image in between


def imshow_torch(name, img):
    if not SHOW_VIS:
        return

    img_cv2 = img.permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    if img_cv2.shape[-1] == 3:
        imshow_cv2(name, img_cv2[..., ::-1])
    elif img_cv2.shape[-1] == 6:
        img_0 = img_cv2[..., :3]
        img_1 = img_cv2[..., 3:]
        imshow_cv2(f"{name}_0", img_0[..., ::-1])
        imshow_cv2(f"{name}_1", img_1[..., ::-1])
    else:
        assert False, f"unknown shape: {img.shape}"


def imshow_cv2(name, img):
    if not SHOW_VIS:
        return

    imsize = img.shape[0]

    if img.shape[0] < 1024:
        img = cv2.resize(
            img,
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imshow(name, img)
    if WRITE_IMG_TMP:
        folder = os.path.join(WRITE_FOLDER, str(imsize), name)
        os.makedirs(folder, exist_ok=True)

        write_idx = WRITE_IDX[name]

        if write_idx % 10 == 0:
            cv2.imwrite(os.path.join(folder, f"{name}-{write_idx:04d}.png"), img)
        WRITE_IDX[name] += 1


def distance_with_batch_dim(a, b):
    return torch.pow(a - b, 2).sum(dim=1)


def squared_l2_distance_with_batch_dim(q, p):
    return distance_with_batch_dim(q.unsqueeze(-1), p.unsqueeze(-2))


def l2_dist(a, b):
    return distance_with_batch_dim(a, b).sum().item()


def low_pass_filter(img, filter_size_x, filter_size_y, border_type_x, border_type_y):

    blurred_x = kornia.filters.gaussian_blur2d(
        img.unsqueeze(0),
        kernel_size=(1, filter_size_x),
        sigma=(filter_size_y, filter_size_x),
        border_type=border_type_x,
    ).squeeze(0)

    blurred_xy = kornia.filters.gaussian_blur2d(
        blurred_x.unsqueeze(0),
        kernel_size=(filter_size_y, 1),
        sigma=(filter_size_y, filter_size_x),
        border_type=border_type_y,
    ).squeeze(0)

    return blurred_xy


@functools.cache
def get_permutations(device):
    perms = torch.tensor(list(itertools.permutations(range(4))), device=device)
    # (perms=24, 4, 4)
    perms_one_hot = torch.nn.functional.one_hot(perms, num_classes=4).float()
    return perms, perms_one_hot


def solve_assignments_batch_dim(cand_dists, device):
    # permutations of assigning 4 elements to 4 positions
    # perms: (perms=24, 4)
    # perms_one_hot: (perms=24, 4, 4)
    perms, perms_one_hot = get_permutations(device)

    # (samples, perms=24)
    perms_dist = torch.einsum("bsce,pce->bsp", cand_dists, perms_one_hot)

    # (samples) [0...23]
    best_perms = torch.argmin(perms_dist, dim=-1)

    # (samples, 4) [0...3]
    best_perms_idx = perms[best_perms]

    return best_perms_idx


def debug_show_blocks(tensor_bchw):
    if not SHOW_VIS:
        return

    # np.random.seed(42)

    num_blocks = tensor_bchw.shape[0]

    print(f"{num_blocks=}")

    # (blocks, _, rgb=3)
    block_colors = torch.tensor(
        np.random.uniform(0, 255, size=(num_blocks, 3)).astype(float)
    ).unsqueeze(-1)

    tensor_bchw[:] = block_colors

    return tensor_bchw


def divisors(n):
    return sorted(
        list(
            set(
                itertools.chain.from_iterable(
                    (i, n // i) for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0
                )
            )
        )
    )


# @torch.compile(fullgraph=True)
def params_to_blocky(
    params, block_size, block_divisor, num_pixel_blocks, shift_y, shift_x
):
    # params: (c, h, w)
    params_rolled = torch.roll(params, (shift_y, shift_x), dims=(1, 2))

    params_truncated = params_rolled[
        :, : num_pixel_blocks * block_size, : num_pixel_blocks * block_size
    ]

    # get the channel out of the way temporarily
    # (c, _, h, w)
    params_c_hw = params_truncated.unsqueeze(1)

    # (c, r^2, h/r, w/r)
    params_unshuffled_cbhw = torch.nn.PixelUnshuffle(block_divisor)(params_c_hw)

    # (r^2, c, h/r, w/r)
    params_unshuffled = params_unshuffled_cbhw.permute(1, 0, 2, 3)

    # (r^2, c, pb, h/r/pb, pb, w/r/pb)
    params_unshuffled_blocky_inline = params_unshuffled.reshape(
        -1,
        params_unshuffled.shape[1],
        num_pixel_blocks,
        block_size,
        num_pixel_blocks,
        block_size,
    )

    # (r^2, pb, pb, c, h/r/pb, w/r/pb)
    params_unshuffled_blocky_first = params_unshuffled_blocky_inline.permute(
        0, 2, 4, 1, 3, 5
    )

    # flatten the batch dimensions
    # (r^2 * pb^2, c, h/r/pb, w/r/pb)
    # TODO replace with flatten?
    params_blocky_batch_flat = params_unshuffled_blocky_first.reshape(
        -1,
        params_unshuffled_blocky_first.shape[3],
        params_unshuffled_blocky_first.shape[4],
        params_unshuffled_blocky_first.shape[5],
    )

    # (r^2 * pb^2, c, block_size * block_size)
    params_blocky_flat = params_blocky_batch_flat.flatten(start_dim=2)

    return params_rolled, params_blocky_flat


# @torch.compile(fullgraph=True)
def blocky_to_params(
    params_rolled,
    params_blocky_flat,
    block_size,
    block_divisor,
    num_pixel_blocks,
    shift_y,
    shift_x,
):
    # unflatten the block dimensions
    # (r^2, pb, pb, c, h/r/pb, w/r/pb)
    params_unshuffled_blocky_first = params_blocky_flat.reshape(
        block_divisor**2,
        num_pixel_blocks,
        num_pixel_blocks,
        params_blocky_flat.shape[1],
        block_size,
        block_size,
    )

    # (r^2, c, pb, h/r/pb, pb, w/r/pb)
    params_unshuffled_blocky_inline = params_unshuffled_blocky_first.permute(
        0, 3, 1, 4, 2, 5
    )

    # (r^2, c, h/r, w/r)
    params_unshuffled = params_unshuffled_blocky_inline.reshape(
        params_unshuffled_blocky_inline.shape[0],
        params_unshuffled_blocky_inline.shape[1],
        params_unshuffled_blocky_inline.shape[2] * block_size,
        params_unshuffled_blocky_inline.shape[4] * block_size,
    )

    # (c, r^2, h/r, w/r)
    params_unshuffled_cbhw = params_unshuffled.permute(1, 0, 2, 3)

    # shuffle the pixels back
    # (c, 1, h, w)
    params_shuffled = torch.nn.PixelShuffle(block_divisor)(params_unshuffled_cbhw)

    params_unshuffled = params_shuffled.squeeze(1)

    params_rolled[
        :, : num_pixel_blocks * block_size, : num_pixel_blocks * block_size
    ] = params_unshuffled

    params = torch.roll(params_rolled, (-shift_y, -shift_x), dims=(1, 2))

    return params


# @torch.compile(options={"epilogue_fusion": True, "max_autotune": True})
def reorder_blocky_shuffled(
    params_blocky_flat, grid_indices_blocky_flat, target_blocky_flat, block_size
):

    shuffled_block_indices = torch.randperm(
        block_size * block_size, device=params_blocky_flat.device
    )
    # shuffled_block_indices = torch.arange(
    #     block_size * block_size, device=params.device
    # )

    # put them in groups of 4
    shuffled_block_indices_cand = shuffled_block_indices.reshape(-1, 4)

    # retrieve params
    blockwise_shuffled_params = params_blocky_flat[:, :, shuffled_block_indices_cand]
    blockwise_shuffled_target = target_blocky_flat[:, :, shuffled_block_indices_cand]

    C = squared_l2_distance_with_batch_dim(
        blockwise_shuffled_params, blockwise_shuffled_target
    )

    # (num_blocks, solver_groups, 4)
    best_perm_indices = solve_assignments_batch_dim(C, device=target_blocky_flat.device)

    # (expanded: num_blocks, solver_groups, 4)
    shuffled_block_indices_cand_exp = shuffled_block_indices_cand.expand_as(
        best_perm_indices
    )

    # best_positions: (num_blocks, solver_groups, 4)
    best_positions = torch.gather(
        shuffled_block_indices_cand_exp, -1, best_perm_indices
    )

    # (num_blocks, solver_groups * 4 == block_size * block_size)
    best_positions_flat = best_positions.flatten(start_dim=1)

    # expands the color dimension
    best_positions_exp = best_positions_flat.unsqueeze(1).expand_as(params_blocky_flat)

    # (num_blocks, c, block_size * block_size)
    best_values = torch.gather(params_blocky_flat, -1, best_positions_exp)
    best_grid_indices = torch.gather(
        grid_indices_blocky_flat, -1, best_positions_flat.unsqueeze(1)
    )

    # (block_size * block_size)
    shuffled_block_indices_cand_flat = shuffled_block_indices_cand.flatten(start_dim=0)

    params_blocky_flat[:, :, shuffled_block_indices_cand_flat] = best_values
    grid_indices_blocky_flat[:, :, shuffled_block_indices_cand_flat] = best_grid_indices

    return params_blocky_flat, grid_indices_blocky_flat.contiguous()


# @torch.compile
def reorder_plas(
    params,
    grid_indices,
    min_block_size,
    filter_size_x,
    filter_size_y,
    border_type_x,
    border_type_y,
    improvement_break,
    pbar,
):
    # Filter the map vectors using the actual filter radius
    target = low_pass_filter(
        params, filter_size_x, filter_size_y, border_type_x, border_type_y
    )

    sidelen = params.shape[1]

    block_size = filter_size_x + 1

    block_size = min(block_size, sidelen)

    # it must be possible to form groups of 4 pixels
    block_size = block_size // 2 * 2
    block_size = max(block_size, min_block_size)

    # IMPORTANT this completely disables the pixel unshuffling
    # leading to all operations being performed on blocks, no strides
    # TODO could get rid of all the pixel unshuffle code, simplify a lot
    # 231109: the new divisor code that allows for non-multiples of min block size size
    # probably breaks this anyway?

    block_divisor = 1

    num_pixel_blocks = sidelen // block_size

    if pbar:
        pbar.set_description(f"filter_size={filter_size_x} - {block_size=}")

    num_reorders = 0

    # for block_divisor in block_divisors:
    # for block_configs in range(5):
    block_config = 0
    while True:

        shift_y = np.random.randint(0, block_size)
        shift_x = np.random.randint(0, block_size)

        # (r^2 * pb^2, c, block_size * block_size)
        params_rolled, params_blocky_flat = params_to_blocky(
            params, block_size, block_divisor, num_pixel_blocks, shift_y, shift_x
        )

        target_rolled, target_blocky_flat = params_to_blocky(
            target, block_size, block_divisor, num_pixel_blocks, shift_y, shift_x
        )

        # TODO perf: could only map the indices here
        # and then index params and target with the mapped indices
        grid_indices_rolled, grid_indices_blocky_flat = params_to_blocky(
            grid_indices, block_size, block_divisor, num_pixel_blocks, shift_y, shift_x
        )

        # replace all values in one block with the same color
        # params_blocky_flat = debug_show_blocks(params_blocky_flat)

        prev_dist = l2_dist(params_blocky_flat, target_blocky_flat)

        # for i in range(n_block_reorders):
        i = 0
        has_improved = False
        while True:

            num_reorders += 1

            # all indices of pixels in the block, shuffled
            params_blocky_flat, grid_indices_blocky_flat = reorder_blocky_shuffled(
                params_blocky_flat,
                grid_indices_blocky_flat,
                target_blocky_flat,
                block_size,
            )

            cur_dist = l2_dist(params_blocky_flat, target_blocky_flat)

            if prev_dist == 0:
                improvement_factor = 0
            else:
                improvement_factor = 1 - (cur_dist / prev_dist)

            if pbar:
                pbar.set_postfix(
                    {
                        "it": "{:04d}".format(i),
                        "dist": f"{cur_dist:.2E}",
                        "dist_factor": f"{improvement_factor:+.2E}",
                    }
                )

            if improvement_factor < improvement_break:
                # print(f"breaking at {filter_size_x=} - {i=} - {improvement_factor=}")
                break

            prev_dist = cur_dist
            i += 1
            has_improved = True

        # (c, h, w)
        params = blocky_to_params(
            params_rolled,
            params_blocky_flat,
            block_size,
            block_divisor,
            num_pixel_blocks,
            shift_y,
            shift_x,
        ).contiguous()

        # TODO perf: could only map the indices here
        grid_indices = blocky_to_params(
            grid_indices_rolled,
            grid_indices_blocky_flat,
            block_size,
            block_divisor,
            num_pixel_blocks,
            shift_y,
            shift_x,
        ).contiguous()

        if SHOW_VIS:
            imshow_torch("target", target)
            imshow_torch("params", params)
            cv2.waitKey(1)

        # ensure that there are a few different configs tried before giving up
        # (for rolling / border improvement)
        if not has_improved and block_config >= 3:
            break

        block_config += 1

    return params, grid_indices, num_reorders


def radius_seq(max_radius, min_radius, radius_update):
    radius = max_radius
    while True:
        yield int(radius)
        radius *= radius_update
        if radius < min_radius:
            break


def sort_with_plas(
    params,
    min_block_size=16,
    min_blur_radius=1,
    improvement_break=1e-5,
    border_type_x="circular",
    border_type_y="reflect",
    seed=None,
    verbose=False,
):
    """ Sorts a set of parameters in a 2xn grid using the Parallel Linear Assignment Sorting (PLAS) algorithm.

    Args:
        border_type_x/y (str): Border for the Gaussian blur that is performed to create the targets for sorting.
                               The expected modes are: 'constant', 'reflect', 'replicate' or 'circular' (kornia gaussian_blur2d border_type).
                               x defaults to 'circular', y defaults to 'reflect': this allows for seamless resampling 1D data into square 2D grids.
        min_blur_radius: Last/smallest blur radius to apply before stopping sort. Defaults to 1 for optimal sort. Increase for earlier stops.
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    grid_shape = params.shape[1:]
    H, W = grid_shape

    # not implemented for non-square aspect ratios
    assert H == W

    start_time = time.time()

    radius_f = max(H, W) / 2 - 1
    radii = list(radius_seq(max_radius=radius_f, min_radius=min_blur_radius, radius_update=0.95))

    if verbose:
        pbar = tqdm(radii)
    else:
        pbar = None

    total_num_reorders = 0

    with torch.inference_mode():

        grid_indices = (
            torch.arange(0, H * W, dtype=torch.int32, device=params.device)
            .reshape(grid_shape)
            .unsqueeze(0)
        )

        for radius in radii:
            # compute filtersize that is smaller than any side of the grid
            filter_size_x = min(W - 1, int(2 * radius + 1))
            filter_size_y = min(H - 1, int(2 * radius + 1))

            params, grid_indices, num_reorders = reorder_plas(
                params,
                grid_indices,
                min_block_size,
                filter_size_x,
                filter_size_y,
                border_type_x,
                border_type_y,
                improvement_break,
                pbar=pbar,
            )

            total_num_reorders += num_reorders

            if pbar:
                pbar.update(1)

    duration = time.time() - start_time

    if verbose:
        print(
            f"\nSorted {params.shape[2]}x{params.shape[2]}={params.shape[1] * params.shape[2]} Gaussians @ {params.shape[0]} dimensions with PLAS in {duration:.3f} seconds \n       with {total_num_reorders} reorders at a rate of {total_num_reorders / duration:.3f} reorders per second"
        )

    return params, grid_indices

#########################################以上是plas相关代码#########################################
import torch.nn.functional as F
from einops import rearrange
import math
def _apply_plas_sorting(squeezed_tensor,plas_min_block_size,plas_min_blur_radius,plas_improvement_break):
        """保持 padding 直到反向完成"""
        T, C, H_sq, W_sq = squeezed_tensor.shape
        H_sq_orig, W_sq_orig = H_sq, W_sq
        
        # 保证输入是方形
        if H_sq != W_sq:
            max_side = max(H_sq, W_sq)
            pad_h = max_side - H_sq
            pad_w = max_side - W_sq
            squeezed_tensor = F.pad(squeezed_tensor, [0, pad_w, 0, pad_h])
            H_sq = W_sq = max_side
        else:
            pad_h = pad_w = 0
        
        # 保证输入可被plas核整除
        block_size = plas_min_block_size
        H_blocks = (H_sq + block_size - 1) // block_size
        W_blocks = (W_sq + block_size - 1) // block_size
        H_needed = H_blocks * block_size
        W_needed = W_blocks * block_size
        
        extra_pad_h = H_needed - H_sq
        extra_pad_w = W_needed - W_sq
        
        if extra_pad_h > 0 or extra_pad_w > 0:
            squeezed_tensor = F.pad(squeezed_tensor, [0, extra_pad_w, 0, extra_pad_h])
            H_sq = H_needed
            W_sq = W_needed
        
        ref_frame = squeezed_tensor[0]
        
        print(f"\n[PLAS] Input shape: {ref_frame.shape}")
        
        # PLAS 排序
        with torch.inference_mode():
            sorted_ref, grid_indices = sort_with_plas(
                ref_frame,
                min_block_size=block_size,
                min_blur_radius=plas_min_blur_radius,
                improvement_break=plas_improvement_break,
                border_type_x='circular',
                border_type_y='circular',
                seed=42,
                verbose=False
            )
        
        # 对所有帧应用相同排序
        indices_flat = grid_indices.flatten().to(torch.int64)
        squeezed_flat = rearrange(squeezed_tensor, 't c h w -> t c (h w)')
        sorted_flat = torch.gather(
            squeezed_flat, 
            dim=2, 
            index=indices_flat.unsqueeze(0).unsqueeze(0).expand(T, C, -1)
        )
        sorted_tensor = rearrange(sorted_flat, 't c (h w) -> t c h w', h=H_sq, w=W_sq)
        
        # ⚠️ 关键：不要裁剪！保持完整的 padded 尺寸
        print(f"[PLAS] Output shape (keeping padding): {sorted_tensor.shape}"))
             
        return sorted_tensor, grid_indices, (H_sq_orig, W_sq_orig, H_sq, W_sq)



def _reverse_plas_sorting(sorted_tensor, grid_indices, size_info):
    """反向时在完整 padded 尺寸上操作"""
    H_sq_orig, W_sq_orig, H_padded, W_padded = size_info
    T, C, H_cur, W_cur = sorted_tensor.shape
    
    print(f"\n[Reverse PLAS] Input shape: {sorted_tensor.shape}")
    print(f"[Reverse PLAS] Expected padded size: ({H_padded}, {W_padded})")
    print(f"[Reverse PLAS] Original size: ({H_sq_orig}, {W_sq_orig})")
    
    # 验证尺寸匹配
    assert H_cur == H_padded and W_cur == W_padded, \
        f"Size mismatch! Current: ({H_cur}, {W_cur}), Expected: ({H_padded}, {W_padded})"
    
    # 在完整的 padded 尺寸上进行反向映射
    indices_flat = grid_indices.flatten().to(torch.int64)
    reverse_indices = torch.zeros_like(indices_flat)
    reverse_indices[indices_flat] = torch.arange(
        len(indices_flat), 
        device=indices_flat.device, 
        dtype=torch.int64
    )
    
    print(f"[Reverse PLAS] Indices range: {indices_flat.min()}-{indices_flat.max()}")
    print(f"[Reverse PLAS] Total pixels: {len(indices_flat)}")
    
    sorted_flat = rearrange(sorted_tensor, 't c h w -> t c (h w)')
    unsorted_flat = torch.gather(
        sorted_flat,
        dim=2,
        index=reverse_indices.unsqueeze(0).unsqueeze(0).expand(T, C, -1)
    )
    unsorted_tensor = rearrange(
        unsorted_flat,
        't c (h w) -> t c h w',
        h=H_padded, w=W_padded
    )
    # ⚠️ 现在才裁剪到原始尺寸
    unsorted_tensor = unsorted_tensor[:, :, :H_sq_orig, :W_sq_orig]
    print(f"[Reverse PLAS] After cropping to original: {unsorted_tensor.shape}")

    return unsorted_tensor

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

        # 3. 保留 patch
        kept = patches[:, mask.squeeze(0)]  # (T, N_keep, p*p*C)

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
        squeezed, plas_indices, plas_pad_info = _apply_plas_sorting(squeezed)
        return squeezed, patched_shape ,plas_indices, plas_pad_info, N_keep

def unsqueeze(mask=None,
            squeezed_tensor = None,
            N_keep=None,
            patch_size=8,
            patched_shape=None,
            region_shape = None,
            plas_indices=None,
            plas_pad_info=None
            ):
    """
    恢复原图（与 StaticRandomMasker 完全一致）
    """
    if mask is None:
        raise RuntimeError("Mask not available")
    squeezed_tensor = _reverse_plas_sorting(
            squeezed_tensor, 
            plas_indices, 
            plas_pad_info
        )
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
    full[:, mask.squeeze(0)] = valid

    # 4. 重组成原图
    recon = rearrange(
        full,
        't (h w) (p1 p2 c) -> t c (h p1) (w p2)',
        h=patched_shape[0], w=patched_shape[1], p1=p, p2=p, c=C
    )

    return recon[:, :, :region_shape[0], :region_shape[1]]
    


