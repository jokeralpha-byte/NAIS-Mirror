import os
import sys
import cv2
import time
import argparse
import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from einops import rearrange

# 导入必要的模块
from module.utils.video_process import read_video_to_array as read_video
from model.super_resolution.span.span import SPAN
from model.inpainter.e2fgvi_hq import InpaintGenerator
from module.masker.masker_for_test import (
    RandomDynamicMaskSyetem, GridMasker, RandomGridMasker, FixedMaskerSystem
)
from module.compressor.compress_with_h265 import (
    _compress_to_h264, _decompress_from_h264,
    _compress_to_h265, _decompress_from_h265
)
from module.compressor.frame_rule import squeeze_frame_independent, unsqueeze_frame_independent
from module.compressor.metadata_compress import save_metadata, load_metadata
from model.inpainter.e2fgvi_hq import InpaintGenerator

try:
    from piqa import SSIM, MS_SSIM, LPIPS
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class MaskBasedViewportProcessor:
    """
    Mask-based Viewport处理器
    流程：降采样 → mask → squeeze → H.264 → unsqueeze → inpaint → SR
    """
    def __init__(
        self, 
        sr_model,
        mask_ratio=0.5,
        h264_crf=23,
        patch_size=16,
        viewport_scale=0.5,
        sr_scale=2,
        device='cuda'
    ):
        self.device = device
        self.sr_model = sr_model
        self.mask_ratio = mask_ratio
        self.h264_crf = h264_crf
        self.patch_size = patch_size
        self.viewport_scale = viewport_scale
        self.sr_scale = sr_scale
        
        # 初始化组件
        self._init_masker()
        self._init_inpainter()
        
        # 临时文件夹
        self.temp_dir = tempfile.mkdtemp()
        
        print(f"[green]✅ MaskBasedViewportProcessor initialized[/green]")
        print(f"  Mask ratio: {mask_ratio}")
        print(f"  H264 CRF: {h264_crf}")
        print(f"  Patch size: {patch_size}")
        print(f"  Viewport scale: {viewport_scale}")
        print(f"  SR scale: {sr_scale}")
    
    def _init_masker(self):
        """初始化Random Grid Masker"""
        self.masker = RandomGridMasker(
            patch_size=self.patch_size
        ).to(self.device).eval()
        print("  [green]✓ Loaded Random Grid Masker[/green]")
    
    def _init_inpainter(self):
        """初始化Inpainter"""
        inpaint_checkpoint = '/data2/mmvisitor/Jia_Daiang/MaStreaming/mastreaming_master/checkpoints/inpainter/E2FGVI-HQ-CVPR22.pth'
        self.inpainter = InpaintGenerator().to(self.device)
        self.inpainter.load_state_dict(
            torch.load(inpaint_checkpoint, map_location=self.device, weights_only=False)
        )
        self.inpainter.eval()
        for p in self.inpainter.parameters():
            p.requires_grad = False
        print("  [green]✓ Loaded inpainter[/green]")
    
    def process(self, viewport):
        """
        处理viewport（整体处理）
        Args:
            viewport: (T, C, H, W) torch.Tensor
        Returns:
            reconstructed: (T, C, H*sr_scale, W*sr_scale) torch.Tensor
            compressed_size: int
        """
        with torch.no_grad():
            # 转换数据类型
            if viewport.dtype == torch.uint8:
                viewport = viewport.float() / 255.0
            elif viewport.dtype != torch.float32:
                viewport = viewport.float()
            
            viewport = viewport.clamp(0, 1).to(self.device)
            
            T, C, H, W = viewport.shape
            print(f"      Input viewport: {viewport.shape}")
            
            # ====== 步骤1: 降采样 ======
            print("    [cyan]→ Downsampling viewport...[/cyan]")
            viewport_down = F.interpolate(
                viewport,
                scale_factor=self.viewport_scale,
                mode='bilinear',
                align_corners=False
            ).clamp(0, 1)
            print(f"      Downsampled: {viewport_down.shape}")
            
            # ====== 步骤2: Masking ======
            print("    [cyan]→ Generating mask...[/cyan]")
            mask_pred = self._generate_mask(viewport_down)
            
            # ====== 步骤3: Squeeze ======
            print("    [cyan]→ Squeezing...[/cyan]")
            squeezed_chunks, metadata = self._squeeze_frames(viewport_down, mask_pred)
            print(f"      Squeezed shape: {squeezed_chunks.shape}")
            
            # ====== 步骤4: H.264压缩 ======
            print("    [cyan]→ H.264 compressing...[/cyan]")
            h264_size = self._compress_squeezed(squeezed_chunks)
            print(f"      H.264: {h264_size/(1024**2):.3f}MB")
            
            # ====== 步骤5: 元数据压缩 ======
            metadata_size = self._compress_metadata(metadata)
            print(f"      Metadata: {metadata_size/(1024**2):.3f}MB")
            
            # 总压缩大小
            compressed_size = h264_size + metadata_size
            print(f"    [green]✓ Total viewport compressed: {compressed_size/(1024**2):.2f}MB[/green]")
            
            # ====== 解压过程 ======
            print("    [cyan]→ Decompressing H.264...[/cyan]")
            decompressed_chunks = self._decompress_squeezed()
            
            print("    [cyan]→ Unsqueezing...[/cyan]")
            unsqueezed = self._unsqueeze_frames(decompressed_chunks, mask_pred, metadata)
            unsqueezed = unsqueezed.clamp(0, 1)
            print(f"      Unsqueezed: {unsqueezed.shape}")
            
            print("    [cyan]→ Inpainting...[/cyan]")
            mask_pixels = self._mask_to_pixels(mask_pred, unsqueezed.shape)
            inpainted = self._inpaint_frames(unsqueezed, mask_pixels)
            inpainted = inpainted.clamp(0, 1)
            print(f"      Inpainted: {inpainted.shape}")
            
            print("    [cyan]→ Super-resolution...[/cyan]")
            sr_output = self._super_resolution(inpainted)
            reconstructed = sr_output.clamp(0, 1)
            print(f"    [green]✓ Reconstruction done: {reconstructed.shape}[/green]")
            
            return reconstructed, compressed_size
    
    def _generate_mask(self, viewport_down):
        """生成Random Grid Mask"""
        T, C, H, W = viewport_down.shape
        viewport_5d = viewport_down.unsqueeze(0)
        mask_pred = self.masker(viewport_5d, self.mask_ratio)
        mask_pred = mask_pred[0]  # (T, H_p, W_p)
        
        kept_ratio = (1 - mask_pred.float().mean().item()) * 100
        print(f"      Mask: {kept_ratio:.1f}% patches kept")
        return mask_pred
    
    def _squeeze_frames(self, viewport_down, mask_pred):
        """Squeeze操作"""
        squeezed_chunks, metadata = squeeze_frame_independent(
            patch_size=self.patch_size,
            mask=~mask_pred,
            pic_tensor=viewport_down
        )
        return squeezed_chunks, metadata
    
    def _compress_squeezed(self, squeezed_chunks):
        """H.264压缩"""
        compressed_bytes = _compress_to_h264(squeezed_chunks, crf=self.h264_crf)
        compressed_size = len(compressed_bytes)
        h264_path = os.path.join(self.temp_dir, 'squeezed.h264')
        with open(h264_path, 'wb') as f:
            f.write(compressed_bytes)
        return compressed_size
    
    def _compress_metadata(self, metadata):
        """压缩元数据"""
        metadata_path = os.path.join(self.temp_dir, 'metadata.pkl.bz2')
        save_metadata(metadata, metadata_path)
        compressed_size = Path(metadata_path).stat().st_size
        return compressed_size
    
    def _decompress_squeezed(self):
        """解压H.264"""
        h264_path = os.path.join(self.temp_dir, 'squeezed.h264')
        with open(h264_path, 'rb') as f:
            compressed_bytes = f.read()
        decompressed_chunks = _decompress_from_h264(compressed_bytes)
        return decompressed_chunks
    
    def _unsqueeze_frames(self, decompressed_chunks, mask_pred, metadata):
        """Unsqueeze操作"""
        metadata_path = os.path.join(self.temp_dir, 'metadata.pkl.bz2')
        metadata = load_metadata(metadata_path)
        unsqueezed = unsqueeze_frame_independent(
            mask=~mask_pred,
            squeezed_tensor=decompressed_chunks,
            metadata=metadata
        )
        if unsqueezed.ndim == 5:
            unsqueezed = unsqueezed[0]
        return unsqueezed
    
    def _mask_to_pixels(self, mask_pred, target_shape):
        """将patch-level mask转换为pixel-level mask"""
        T, H_p, W_p = mask_pred.shape
        mask_exp = mask_pred.unsqueeze(3).expand(-1, -1, -1, self.patch_size * self.patch_size * 3)
        mask_pixels = rearrange(
            mask_exp,
            't h_p w_p (c p1 p2) -> t c (h_p p1) (w_p p2)',
            p1=self.patch_size, 
            p2=self.patch_size,
            c=3
        ).to(self.device)
        return mask_pixels.float()
    
    def _inpaint_frames(self, frames, mask_pixels):
        """Inpainting"""
        T, C, H, W = frames.shape
        
        # 归一化到[-1, 1]
        frames = frames * 2.0 - 1.0
        mask_pixels = 1 - mask_pixels
        masked_frames = frames * mask_pixels
        masked_frames = masked_frames.unsqueeze(0)  # (1, T, C, H, W)
        
        # Padding到inpainter需要的尺寸
        mod_size_h, mod_size_w = 60, 108
        h_pad = (mod_size_h - H % mod_size_h) % mod_size_h
        w_pad = (mod_size_w - W % mod_size_w) % mod_size_w
        
        # 使用镜像padding
        padded_imgs = torch.cat([masked_frames, torch.flip(masked_frames, [3])], 3)[:, :, :, :H + h_pad, :]
        B, T_pad, C_pad, H_pad, W_pad = padded_imgs.shape
        padded_imgs = torch.cat([padded_imgs, torch.flip(padded_imgs, [4])], 4)[:, :, :, :, :W + w_pad]
        
        # Inpaint
        with torch.no_grad():
            pred_imgs, _ = self.inpainter(padded_imgs, T)
        
        # 重排并裁剪
        pred_imgs = rearrange(pred_imgs, '(b t) c h w -> b t c h w', b=B, t=T)
        pred_imgs = pred_imgs[:, :, :, :H, :W]
        pred_imgs = (pred_imgs + 1.0) / 2.0
        
        # 融合
        mask_pixels = mask_pixels.unsqueeze(0)
        frames = (frames + 1.0) / 2.0
        inpainted = frames.unsqueeze(0) * mask_pixels + pred_imgs * (1 - mask_pixels)
        inpainted = inpainted.clamp(0, 1)
        inpainted = inpainted[0]  # (T, C, H, W)
        
        return inpainted
    
    def _super_resolution(self, frames):
        """超分辨率"""
        with torch.no_grad():
            sr_output = self.sr_model(frames)
        return sr_output
    
    def __del__(self):
        """清理临时文件"""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def load_sr_model(checkpoint_path, device, sr_scale=2):
    """加载超分辨率模型"""
    print("[cyan]Loading SR model...[/cyan]")
    sr_model = SPAN(3, 3, upscale=sr_scale, feature_channels=48).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    sr_model.load_state_dict(ckpt['params_ema'])
    sr_model.eval()
    for p in sr_model.parameters():
        p.requires_grad = False
    print("[green]✓ SR model loaded[/green]")
    return sr_model


def calculate_viewport_config(full_resolution, target_viewport_size=1280):
    """计算合适的viewport配置"""
    H, W = full_resolution
    
    best_config = None
    best_diff = float('inf')
    
    for NH in [6, 8, 10, 12]:
        for NW in [6, 8, 10, 12]:
            tile_h = H // NH
            tile_w = W // NW
            
            for Nh in range(2, NH):
                for Nw in range(2, NW):
                    viewport_h = Nh * tile_h
                    viewport_w = Nw * tile_w
                    
                    diff = abs(viewport_h - target_viewport_size) + abs(viewport_w - target_viewport_size)
                    
                    aspect_ratio = max(viewport_h, viewport_w) / min(viewport_h, viewport_w)
                    if aspect_ratio < 1.5 and diff < best_diff:
                        best_diff = diff
                        best_config = ((NH, NW), (Nh, Nw), (viewport_h, viewport_w))
    
    if best_config:
        tile_grid, viewport_tiles, viewport_size = best_config
        print(f"[green]Calculated viewport config:[/green]")
        print(f"  Tile grid: {tile_grid[0]}x{tile_grid[1]}")
        print(f"  Viewport tiles: {viewport_tiles[0]}x{viewport_tiles[1]}")
        print(f"  Viewport size: {viewport_size[0]}x{viewport_size[1]}")
        return tile_grid, viewport_tiles
    else:
        return (8, 8), (4, 4)


def select_viewport_tiles(frame_shape, tile_grid, viewport_tiles, center_lat=0, center_lon=0):
    """选择viewport的tiles"""
    NH, NW = tile_grid
    Nh, Nw = viewport_tiles
    H, W = frame_shape
    
    tile_h = H // NH
    tile_w = W // NW
    
    center_x = int((center_lon + 180) / 360 * W)
    center_y = int((90 - center_lat) / 180 * H)
    
    center_tile_i = center_y // tile_h
    center_tile_j = center_x // tile_w
    
    start_i = max(0, min(center_tile_i - Nh // 2, NH - Nh))
    start_j = center_tile_j - Nw // 2
    
    selected_indices = []
    for i in range(start_i, start_i + Nh):
        if i >= NH:
            continue
        for j in range(start_j, start_j + Nw):
            j_wrapped = j % NW
            selected_indices.append((i, j_wrapped))
    
    return selected_indices


def extract_viewport(frames, selected_indices, tile_grid):
    """从全景帧中提取viewport（完整拼接）"""
    T, C, H, W = frames.shape
    NH, NW = tile_grid
    tile_h = H // NH
    tile_w = W // NW
    
    selected_array = np.array(selected_indices)
    rows = sorted(set(selected_array[:, 0]))
    cols = sorted(set(selected_array[:, 1]))
    
    Nh = len(rows)
    Nw = len(cols)
    
    if frames.dtype == torch.uint8:
        frames = frames.float() / 255.0
    elif frames.dtype != torch.float32:
        frames = frames.float()
    
    frames = frames.clamp(0, 1)
    
    viewport = torch.zeros(T, C, Nh * tile_h, Nw * tile_w, 
                          dtype=torch.float32, device=frames.device)
    
    for idx, (i, j) in enumerate(selected_indices):
        row_in_viewport = rows.index(i)
        col_in_viewport = cols.index(j)
        
        tile = frames[:, :, i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
        viewport[:, :, 
                row_in_viewport*tile_h:(row_in_viewport+1)*tile_h,
                col_in_viewport*tile_w:(col_in_viewport+1)*tile_w] = tile
    
    return viewport


def process_background(background, spatial_scale=0.125, crf=40):
    """处理background（空间降采样+H.265压缩）"""
    T, C, H, W = background.shape
    
    if background.dtype == torch.uint8:
        background = background.float() / 255.0
    elif background.dtype != torch.float32:
        background = background.float()
    
    background = background.clamp(0, 1)
    
    H_lr = int(H * spatial_scale)
    W_lr = int(W * spatial_scale)
    
    downsampled = F.interpolate(
        background,
        size=(H_lr, W_lr),
        mode='bilinear',
        align_corners=False
    ).clamp(0, 1)
    
    frames_to_compress = downsampled[[0, 2, 4]]
    compressed_bytes = _compress_to_h265(frames_to_compress, crf=crf)
    compressed_size = len(compressed_bytes)
    decompressed = _decompress_from_h265(compressed_bytes).clamp(0, 1)
    
    background_lr = torch.zeros_like(downsampled)
    for t in range(T):
        if t < 2:
            background_lr[t] = decompressed[0]
        elif t < 4:
            background_lr[t] = decompressed[1]
        else:
            background_lr[t] = decompressed[2]
    
    upsampled = F.interpolate(
        background_lr,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).clamp(0, 1)
    
    return upsampled, compressed_size


def stitch_viewport_to_background(viewport, background, selected_indices, tile_grid, border_width=5):
    """将viewport拼接回background，并添加边框"""
    stitched = background.clone()
    
    T, C, H, W = background.shape
    NH, NW = tile_grid
    tile_h = H // NH
    tile_w = W // NW
    
    selected_array = np.array(selected_indices)
    rows = sorted(set(selected_array[:, 0]))
    cols = sorted(set(selected_array[:, 1]))
    
    Nh = len(rows)
    Nw = len(cols)
    
    for idx, (i, j) in enumerate(selected_indices):
        row_in_viewport = rows.index(i)
        col_in_viewport = cols.index(j)
        
        viewport_patch = viewport[:, :, 
                                  row_in_viewport*tile_h:(row_in_viewport+1)*tile_h, 
                                  col_in_viewport*tile_w:(col_in_viewport+1)*tile_w]
        
        stitched[:, :, i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = viewport_patch
    
    if border_width > 0:
        border_color = torch.zeros(1, C, 1, 1, device=stitched.device)
        
        min_i, max_i = min(rows), max(rows)
        min_j, max_j = min(cols), max(cols)
        
        viewport_top = min_i * tile_h
        viewport_bottom = (max_i + 1) * tile_h
        viewport_left = min_j * tile_w
        viewport_right = (max_j + 1) * tile_w
        
        wrap_around = len(set(cols)) != (max_j - min_j + 1)
        
        if not wrap_around:
            if viewport_top >= border_width:
                stitched[:, :, max(0, viewport_top-border_width):viewport_top, 
                        viewport_left:viewport_right] = border_color
            if viewport_bottom + border_width <= H:
                stitched[:, :, viewport_bottom:min(H, viewport_bottom+border_width), 
                        viewport_left:viewport_right] = border_color
            
            if viewport_left >= border_width:
                stitched[:, :, viewport_top:viewport_bottom, 
                        max(0, viewport_left-border_width):viewport_left] = border_color
            if viewport_right + border_width <= W:
                stitched[:, :, viewport_top:viewport_bottom, 
                        viewport_right:min(W, viewport_right+border_width)] = border_color
        else:
            for (i, j) in selected_indices:
                if i == min_i and i * tile_h >= border_width:
                    stitched[:, :, max(0, i*tile_h-border_width):i*tile_h,
                            j*tile_w:(j+1)*tile_w] = border_color
                if i == max_i and (i+1) * tile_h + border_width <= H:
                    stitched[:, :, (i+1)*tile_h:min(H, (i+1)*tile_h+border_width),
                            j*tile_w:(j+1)*tile_w] = border_color
                
                if (i, (j-1) % NW) not in selected_indices:
                    if j * tile_w >= border_width:
                        stitched[:, :, i*tile_h:(i+1)*tile_h,
                                max(0, j*tile_w-border_width):j*tile_w] = border_color
                
                if (i, (j+1) % NW) not in selected_indices:
                    if (j+1) * tile_w + border_width <= W:
                        stitched[:, :, i*tile_h:(i+1)*tile_h,
                                (j+1)*tile_w:min(W, (j+1)*tile_w+border_width)] = border_color
    
    return stitched


def load_viewing_trajectory(trajectory_file):
    """加载视角轨迹"""
    trajectory = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                frame_idx = int(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])
                trajectory.append((frame_idx, lat, lon))
    return trajectory


def initialize_metrics(device):
    """初始化质量评估指标"""
    if not METRICS_AVAILABLE:
        return None
    
    metrics = {
        'ssim': SSIM().to(device),
        'ms_ssim': MS_SSIM().to(device),
        'lpips': LPIPS(network='alex').to(device)
    }
    
    for metric in metrics.values():
        metric.eval()
        for p in metric.parameters():
            p.requires_grad = False
    
    return metrics


def compute_metrics(original, reconstructed, metrics):
    """计算质量指标"""
    if metrics is None:
        return {}
    
    with torch.no_grad():
        results = {}
        
        mse = F.mse_loss(reconstructed, original)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
        results['psnr'] = psnr.item()
        
        ssim_val = metrics['ssim'](reconstructed, original)
        results['ssim'] = ssim_val.item()
        
        ms_ssim_val = metrics['ms_ssim'](reconstructed, original)
        results['ms_ssim'] = ms_ssim_val.item()
        
        lpips_val = metrics['lpips'](reconstructed, original)
        results['lpips'] = lpips_val.item()
        
        return results


def save_video_opencv(video_array, output_path, fps=30):
    """使用OpenCV保存视频"""
    T, H, W, C = video_array.shape
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    if not out.isOpened():
        print(f"[yellow]Failed with mp4v, trying alternative codecs...[/yellow]")
        for codec in ['avc1', 'H264', 'X264']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
            if out.isOpened():
                print(f"[green]Using codec: {codec}[/green]")
                break
    
    if not out.isOpened():
        print(f"[red]Error: Could not initialize video writer[/red]")
        return False
    
    failed_frames = 0
    for i in tqdm(range(T), desc="Writing frames"):
        frame = video_array[i]
        
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        success = out.write(frame_bgr)
        if not success:
            failed_frames += 1
    
    out.release()
    
    if failed_frames > 0:
        print(f"[yellow]Warning: {failed_frames}/{T} frames failed to write[/yellow]")
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        if file_size > 0:
            print(f"\n[bold green]✓ Video saved successfully![/bold green]")
            print(f"  Path: {output_path}")
            print(f"  Size: {file_size:.2f} MB")
            print(f"  Resolution: {W}x{H}")
            print(f"  Frames: {T}")
            print(f"  FPS: {fps}")
            return True
        else:
            print(f"[red]✗ Error: Video file is empty[/red]")
            return False
    else:
        print(f"[red]✗ Error: Video file was not created[/red]")
        return False


def process_demo(args, device):
    """主处理流程"""
    
    print("\n" + "="*60)
    print("[bold cyan]Step 1: Loading Video[/bold cyan]")
    print("="*60)
    
    frames_full,_ = read_video(args.video_path, max_frames=args.num_frames)
    frames_full = torch.from_numpy(frames_full)
    fps = 30
    
    if args.num_frames > 0:
        frames_full = frames_full[:args.num_frames]
    
    print(f"[green]✓ Loaded {len(frames_full)} frames[/green]")
    print(f"  Resolution: {frames_full.shape[2]}x{frames_full.shape[3]}")
    print(f"  Data type: {frames_full.dtype}")
    
    print("\n" + "="*60)
    print("[bold cyan]Step 2: Calculating Viewport Configuration[/bold cyan]")
    print("="*60)
    
    H, W = frames_full.shape[2], frames_full.shape[3]
    tile_grid, viewport_tiles = calculate_viewport_config(
        (H, W), 
        target_viewport_size=args.target_viewport_size
    )
    
    print("\n" + "="*60)
    print("[bold cyan]Step 3: Loading Models[/bold cyan]")
    print("="*60)
    
    sr_model = load_sr_model(args.sr_checkpoint, device, sr_scale=args.sr_scale)
    
    viewport_processor = MaskBasedViewportProcessor(
        sr_model=sr_model,
        mask_ratio=args.mask_ratio,
        h264_crf=args.h264_crf,
        patch_size=args.patch_size,
        viewport_scale=args.viewport_scale,
        sr_scale=args.sr_scale,
        device=device
    )
    
    metrics = None
    if args.do_eval:
        print("\n[cyan]Initializing quality metrics...[/cyan]")
        metrics = initialize_metrics(device)
        if metrics:
            print("[green]✓ Metrics initialized[/green]")
    
    print("\n" + "="*60)
    print("[bold cyan]Step 4: Loading Viewing Trajectory[/bold cyan]")
    print("="*60)
    
    if os.path.exists(args.output_angel_dir):
        trajectory = load_viewing_trajectory(args.output_angel_dir)
        print(f"[green]✓ Loaded {len(trajectory)} trajectory points[/green]")
    else:
        trajectory = [(i, 0, 0) for i in range(len(frames_full))]
        print(f"[yellow]No trajectory file, using default (0, 0)[/yellow]")
    
    print("\n" + "="*60)
    print("[bold cyan]Step 5: Processing Video[/bold cyan]")
    print("="*60)
    
    total_frames = len(frames_full)
    chunk_length = args.chunk_length
    num_chunks = (total_frames + chunk_length - 1) // chunk_length
    
    print(f"Total frames: {total_frames}")
    print(f"Chunk length: {chunk_length}")
    print(f"Number of chunks: {num_chunks}\n")
    
    processed_video = []
    total_viewport_size = 0
    total_background_size = 0
    total_original_size = 0
    total_metrics = {'psnr': [], 'ssim': [], 'ms_ssim': [], 'lpips': []}
    
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_length
        end_frame = min(start_frame + chunk_length, total_frames)
        
        print(f"\n{'='*60}")
        print(f"[bold yellow]Chunk {chunk_idx+1}/{num_chunks}[/bold yellow]")
        print(f"  Frames: {start_frame} - {end_frame-1}")
        print(f"{'='*60}\n")
        
        chunk = frames_full[start_frame:end_frame].to(device)
        T_chunk, C, H, W = chunk.shape
        
        original_size = T_chunk * C * H * W * 4
        total_original_size += original_size
        
        print("[cyan]→ Determining viewport position...[/cyan]")
        center_frame_idx = start_frame + T_chunk // 2
        center_lat, center_lon = 0, 0
        for frame_idx, lat, lon in trajectory:
            if frame_idx == center_frame_idx:
                center_lat, center_lon = lat, lon
                break
        print(f"  Center: frame {center_frame_idx}, ({center_lat:.2f}°, {center_lon:.2f}°)")
        
        print("[cyan]→ Selecting viewport tiles...[/cyan]")
        selected_indices = select_viewport_tiles(
            (H, W), tile_grid, viewport_tiles,
            center_lat, center_lon
        )
        print(f"  Selected {len(selected_indices)} tiles")
        
        print("[cyan]→ Extracting viewport...[/cyan]")
        viewport = extract_viewport(chunk, selected_indices, tile_grid)
        print(f"  Viewport shape: {viewport.shape}")
        
        print("[cyan]→ Processing viewport (mask-based)...[/cyan]")
        start_time = time.time()
        processed_viewport, viewport_size = viewport_processor.process(viewport)
        viewport_time = time.time() - start_time
        print(f"  [green]✓ Time: {viewport_time:.2f}s, Size: {viewport_size/(1024**2):.2f}MB[/green]")
        total_viewport_size += viewport_size
        
        print("[cyan]→ Processing background...[/cyan]")
        start_time = time.time()
        processed_background, background_size = process_background(
            chunk, args.background_spatial_scale, args.background_crf
        )
        background_time = time.time() - start_time
        print(f"  [green]✓ Time: {background_time:.2f}s, Size: {background_size/(1024**2):.2f}MB[/green]")
        total_background_size += background_size
        
        print("[cyan]→ Stitching viewport to background...[/cyan]")
        final_chunk = stitch_viewport_to_background(
            processed_viewport.cpu(),
            processed_background.cpu(),
            selected_indices,
            tile_grid,
            border_width=args.border_width
        )
        print(f"  Final shape: {final_chunk.shape}")
        
        if args.do_eval and metrics:
            print("[cyan]→ Computing metrics...[/cyan]")
            chunk_float = chunk.cpu().float()
            if chunk.dtype == torch.uint8:
                chunk_float = chunk_float / 255.0
            chunk_float = chunk_float.clamp(0, 1)
            final_chunk = final_chunk.clamp(0, 1)
            
            chunk_metrics = compute_metrics(
                chunk_float, final_chunk,
                {k: v.cpu() for k, v in metrics.items()}
            )
            
            for k, v in chunk_metrics.items():
                total_metrics[k].append(v)
            
            print(f"  PSNR: {chunk_metrics['psnr']:.2f} dB")
            print(f"  SSIM: {chunk_metrics['ssim']:.4f}")
        
        processed_video.append(final_chunk)
        
        del chunk, viewport, processed_viewport, processed_background, final_chunk
        torch.cuda.empty_cache()
        
        print(f"\n[green]✓ Chunk {chunk_idx+1} completed[/green]")
    
    print("\n" + "="*60)
    print("[bold green]Statistics[/bold green]")
    print("="*60)
    
    print(f"\n[cyan]Configuration:[/cyan]")
    print(f"  Tile grid: {tile_grid[0]}×{tile_grid[1]}")
    print(f"  Viewport tiles: {viewport_tiles[0]}×{viewport_tiles[1]}")
    print(f"  Mask ratio: {args.mask_ratio}")
    print(f"  Viewport scale: {args.viewport_scale}")
    print(f"  SR scale: {args.sr_scale}")
    print(f"  H.264 CRF: {args.h264_crf}")
    
    if args.do_eval and total_metrics['psnr']:
        print("\n[yellow]Quality Metrics:[/yellow]")
        for metric_name, values in total_metrics.items():
            if values:
                avg_value = np.mean(values)
                if 'lpips' in metric_name:
                    print(f"  {metric_name.upper()}: {avg_value:.4f} (lower is better)")
                else:
                    print(f"  {metric_name.upper()}: {avg_value:.4f}")
    
    print("\n[magenta]Compression:[/magenta]")
  
    print(f"  Viewport: {total_viewport_size / (1024**2):.2f} MB")
    print(f"  Background: {total_background_size / (1024**2):.2f} MB")
    print(f"  Total: {(total_viewport_size + total_background_size) / (1024**2):.2f} MB")
    
    
    print("="*60)
    
    print("\n[cyan]Saving video...[/cyan]")
    if processed_video:
        final_video = torch.cat(processed_video, dim=0)
        
        if isinstance(final_video, torch.Tensor):
            final_video = final_video.cpu().numpy()
        
        if final_video.shape[1] == 3:
            final_video = np.transpose(final_video, (0, 2, 3, 1))
        
        if final_video.dtype != np.uint8:
            final_video = (final_video * 255).clip(0, 255).astype(np.uint8)
        
        success = save_video_opencv(final_video, args.output_path, fps)
        
        if not success:
            print("[red]Failed to save video[/red]")
    else:
        print("[red]No chunks processed[/red]")


def main():
    parser = argparse.ArgumentParser(
        description='360° Video Streaming with Mask-based Viewport Processing'
    )
    
    parser.add_argument('--video_path', type=str,
                       default='/data2/mmvisitor/Jia_Daiang/360_video_compression/SpectraMind360/input/冰川冲浪_爱给网_aigei_com.mp4',
                       help='输入360°视频路径')
    parser.add_argument('--sr_checkpoint', type=str,
                       default='/data2/mmvisitor/Jia_Daiang/360_video_compression/SpectraMind360/checkpoint/spanx2_ch48.pth',
                       help='超分模型路径')
    parser.add_argument('--output_path', type=str,
                       default='/data2/mmvisitor/Jia_Daiang/360_video_compression/SpectraMind360/output/output_mask_based.mp4',
                       help='输出视频路径')
    parser.add_argument('--output_angel_dir', type=str,
                       default='/data2/mmvisitor/Jia_Daiang/360_video_compression/SpectraMind360/angels/angels.txt',
                       help='视角轨迹文件')
    
    parser.add_argument('--target_viewport_size', type=int, default=1280,
                       help='目标viewport大小')
    
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                       help='Mask比例')
    parser.add_argument('--h264_crf', type=int, default=23,
                       help='H.264 CRF')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch大小')
    parser.add_argument('--viewport_scale', type=float, default=0.5,
                       help='Viewport降采样比例')
    parser.add_argument('--sr_scale', type=int, default=2,
                       help='超分倍数')
    
    parser.add_argument('--background_spatial_scale', type=float, default=0.125,
                       help='Background降采样比例')
    parser.add_argument('--background_crf', type=int, default=40,
                       help='Background CRF')
    
    parser.add_argument('--chunk_length', type=int, default=5,
                       help='Chunk长度')
    parser.add_argument('--num_frames', type=int, default=60,
                       help='处理帧数（0=全部）')
    
    parser.add_argument('--border_width', type=int, default=8,
                       help='Viewport边框宽度')
    
    parser.add_argument('--do_eval', action='store_true', default=True,
                       help='是否评估质量')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("[bold]360° Video Streaming - Mask-based Method[/bold]")
    print("="*60)
    
    print(f"\n[bold]Configuration:[/bold]")
    print(f"  Device: {device}")
    print(f"  Input: {args.video_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Target viewport size: {args.target_viewport_size}x{args.target_viewport_size}")
    print(f"  Mask ratio: {args.mask_ratio}")
    print(f"  Viewport scale: {args.viewport_scale}")
    print(f"  SR scale: {args.sr_scale}")
    print(f"  Chunk length: {args.chunk_length}")
    print("="*60 + "\n")
    
    process_demo(args, device)


if __name__ == '__main__':
    main()

