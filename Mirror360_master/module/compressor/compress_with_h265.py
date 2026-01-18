

import io
import av
import torch
import torch.nn.functional as F
import numpy as np
def _compress_to_h265(
    video_tensor, 
    fps=30, 
    bitrate=None, 
    crf=28,  # H.265默认值
    intra_only=False,
    verbose=False
):
    """
    H.265压缩 - 实时流媒体配置
    
    🎯 使用场景解释：
    模拟实时视频流传输场景（如视频会议、直播、云游戏等），
    要求：
    1. 编码延迟 < 100ms（单帧）
    2. 支持低延迟解码
    3. 计算资源受限（移动设备、边缘服务器）
    
    因此采用 ultrafast preset 和 zerolatency tune，
    这是实际部署中最常用的配置，但会牺牲15-25%的压缩效率。
    
    参考：
    - FFmpeg官方文档推荐的直播配置
    - WebRTC、Zoom等使用的实时编码策略
    """
    if video_tensor.dim() != 4 or video_tensor.shape[1] != 3:
        raise ValueError("Input tensor must be (T, C, H, W) with C=3.")
    
    T, C, H, W = video_tensor.shape
    if H == 0 or W == 0:
        return b''
    
    # 确保尺寸是偶数
    pad_w = W % 2
    pad_h = H % 2
    if pad_w or pad_h:
        video_tensor = F.pad(video_tensor, (0, pad_w, 0, pad_h))
        H, W = video_tensor.shape[2:]
    
    arr = (video_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    out = io.BytesIO()
    
    with av.open(out, mode='w', format='mp4') as container:
        stream = container.add_stream('libx265', rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = 'yuv420p'
        
        # 🔥 实时流媒体的标准配置
        x265_params = [
            'log-level=0',
            
            # 核心：速度优先配置
            'preset=ultrafast',      # 最快编码速度（~10x faster than medium）
            'tune=zerolatency',      # 零延迟调优（禁用前瞻、减少缓冲）
            
            # 限制计算复杂度
            'ref=1',                 # 只用1个参考帧（默认是3-5个）
            'bframes=0',             # 不使用B帧（减少延迟和复杂度）
            'rc-lookahead=0',        # 关闭前瞻（实时场景无法预知未来帧）
            
            # 简化的运动估计
            'me=dia',                # 最简单的运动估计算法（diamond）
            'subme=0',               # 最快的子像素运动估计
            
            # 简化的率失真优化
            'rd=2',                  # 较低的RD级别（默认是3，最高6）
            
            # 关闭高级特性
            'sao=0',                 # 关闭SAO滤波（节省20-30%编码时间）
            'amp=0',                 # 关闭非对称运动分区
            'rect=0',                # 关闭矩形分区
            
            # 感知优化（保留一些，避免质量太差）
            'aq-mode=1',             # 保留基础自适应量化
            'psy-rd=1.0',            # 适度的感知优化
        ]

        if intra_only:
            x265_params.extend([
                'keyint=1',
                'scenecut=0',
            ])
        
        if bitrate:
            x265_params.append(f'bitrate={bitrate}')
        else:
            x265_params.append(f'crf={crf}')
        
        stream.options = {'x265-params': ':'.join(x265_params)}
        
        for i in range(T):
            frame = av.VideoFrame.from_ndarray(arr[i], format='rgb24')
            for pkt in stream.encode(frame):
                container.mux(pkt)
        
        for pkt in stream.encode():
            container.mux(pkt)
    
    compressed = out.getvalue()
    
    if verbose:
        mode = "Real-time intra-only" if intra_only else "Real-time inter"
        print(f"[H.265-RT] 模式: {mode} | CRF: {crf}")
        print(f"[H.265-RT] 配置: ultrafast preset (实时流媒体标准)")
        print(f"[H.265-RT] 压缩后: {len(compressed)/1024:.2f} KB")
        print(f"[H.265-RT] 说明: 模拟视频会议/云游戏等低延迟场景")
    
    return compressed

def _decompress_from_h265(compressed_bytes, device='cuda'):
    """
    H.265视频解压缩
    
    Args:
        compressed_bytes: 压缩后的字节数据
        device: 目标设备
        
    Returns:
        video_tensor: (T, C, H, W) 视频张量
    """
    if not compressed_bytes:
        return torch.empty(0, 3, 0, 0, device=device)
    
    inp = io.BytesIO(compressed_bytes)
    frames = []
    
    with av.open(inp, mode='r', options={'loglevel': 'error'}) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
    
    if not frames:
        return torch.empty(0, 3, 0, 0, device=device)
    
    arr = np.stack(frames)
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).to(device)
    
    return tensor
def _compress_to_h264(
    video_tensor, 
    fps=30, 
    bitrate=None, 
    crf=23,
    intra_only=False,
    verbose=False
):
    """
    H.264 视频压缩 - 实时模式
    """
    if video_tensor.dim() != 4 or video_tensor.shape[1] != 3:
        raise ValueError("Input tensor must be (T, C, H, W) with C=3.")
    
    T, C, H, W = video_tensor.shape
    if H == 0 or W == 0:
        return b''
    
    # 确保尺寸是偶数
    pad_w = W % 2
    pad_h = H % 2
    if pad_w or pad_h:
        video_tensor = F.pad(video_tensor, (0, pad_w, 0, pad_h))
        H, W = video_tensor.shape[2:]
    
    # 转为 numpy uint8 格式
    arr = (video_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    
    out = io.BytesIO()
    
    try:
        with av.open(out, mode='w', format='mp4') as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width = W
            stream.height = H
            stream.pix_fmt = 'yuv420p'
            
            # 🔥 实时编码参数配置
            x264_params = [
                'preset=ultrafast',      # 最快速度
                'tune=zerolatency',      # 零延迟优化
                'rc-lookahead=0',        # 禁用码率控制前瞻
                'bframes=0',             # 禁用B帧减少延迟
            ]

            if intra_only:
                x264_params.extend([
                    'keyint=1',
                    'min-keyint=1',
                    'scenecut=0',
                ])
            else:
                x264_params.append('keyint=30')  # GOP大小

            # 码率控制
            if bitrate:
                x264_params.append(f'bitrate={bitrate}')
                x264_params.append(f'vbv-maxrate={bitrate}')
                x264_params.append(f'vbv-bufsize={int(bitrate)}')
            else:
                x264_params.append(f'crf={crf}')
            
            stream.options = {'x264-params': ':'.join(x264_params)}
            
            # 编码循环
            for i in range(T):
                frame = av.VideoFrame.from_ndarray(arr[i], format='rgb24')
                for pkt in stream.encode(frame):
                    container.mux(pkt)
            
            # Flush
            for pkt in stream.encode():
                container.mux(pkt)
                
    except Exception as e:
        print(f"[H.264 Error] Encoding failed: {e}")
        return b''
    
    compressed = out.getvalue()
    
    if verbose:
        mode = "Intra-only (I-Frame)" if intra_only else "Inter (IPB)"
        ratio = (T*H*W*3) / (len(compressed) + 1e-6)
        print(f"[H.264] Mode: {mode} | CRF/Bitrate: {crf if not bitrate else bitrate}")
        print(f"[H.264] Size: {len(compressed)/1024:.2f} KB | Ratio: {ratio:.1f}:1")
    
    return compressed

def _decompress_from_h264(compressed_bytes, device='cuda'):
    """
    H.265视频解压缩
    
    Args:
        compressed_bytes: 压缩后的字节数据
        device: 目标设备
        
    Returns:
        video_tensor: (T, C, H, W) 视频张量
    """
    if not compressed_bytes:
        return torch.empty(0, 3, 0, 0, device=device)
    
    inp = io.BytesIO(compressed_bytes)
    frames = []
    
    with av.open(inp, mode='r', options={'loglevel': 'error'}) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
    
    if not frames:
        return torch.empty(0, 3, 0, 0, device=device)
    
    arr = np.stack(frames)
    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2).to(device)
    
    return tensor

def _compress_to_h265_forexpert(
    video_tensor, 
    fps=30, 
    bitrate=None, 
    crf=45,          # 建议不要超过 50，太高有时也会报错
    intra_only=False,
    verbose=False
):
    
        """
        H.265压缩 - 实时流媒体配置
        
        🎯 使用场景解释：
        模拟实时视频流传输场景（如视频会议、直播、云游戏等），
        要求：
        1. 编码延迟 < 100ms（单帧）
        2. 支持低延迟解码
        3. 计算资源受限（移动设备、边缘服务器）
        
        因此采用 ultrafast preset 和 zerolatency tune，
        这是实际部署中最常用的配置，但会牺牲15-25%的压缩效率。
        
        参考：
        - FFmpeg官方文档推荐的直播配置
        - WebRTC、Zoom等使用的实时编码策略
        """
        if video_tensor.dim() != 4 or video_tensor.shape[1] != 3:
            raise ValueError("Input tensor must be (T, C, H, W) with C=3.")
        
        T, C, H, W = video_tensor.shape
        if H == 0 or W == 0:
            return b''
        
        # 确保尺寸是偶数
        pad_w = W % 2
        pad_h = H % 2
        if pad_w or pad_h:
            video_tensor = F.pad(video_tensor, (0, pad_w, 0, pad_h))
            H, W = video_tensor.shape[2:]
        
        arr = (video_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        out = io.BytesIO()
        
        with av.open(out, mode='w', format='mp4') as container:
            stream = container.add_stream('libx265', rate=fps)
            stream.width, stream.height = W, H
            stream.pix_fmt = 'yuv420p'
            
            # 🔥 实时流媒体的标准配置
            x265_params = [
                'log-level=0',
                
                # 核心：速度优先配置
                'preset=ultrafast',      # 最快编码速度（~10x faster than medium）
                'tune=zerolatency',      # 零延迟调优（禁用前瞻、减少缓冲）
                
                # 限制计算复杂度
                'ref=1',                 # 只用1个参考帧（默认是3-5个）
                'bframes=0',             # 不使用B帧（减少延迟和复杂度）
                'rc-lookahead=0',        # 关闭前瞻（实时场景无法预知未来帧）
                
                # 简化的运动估计
                'me=dia',                # 最简单的运动估计算法（diamond）
                'subme=0',               # 最快的子像素运动估计
                
                # 简化的率失真优化
                'rd=2',                  # 较低的RD级别（默认是3，最高6）
                
                # 关闭高级特性
                'sao=0',                 # 关闭SAO滤波（节省20-30%编码时间）
                'amp=0',                 # 关闭非对称运动分区
                'rect=0',                # 关闭矩形分区
                
                # 感知优化（保留一些，避免质量太差）
                'aq-mode=1',             # 保留基础自适应量化
                'psy-rd=1.0',            # 适度的感知优化
            ]

            if intra_only:
                x265_params.extend([
                    'keyint=1',
                    'scenecut=0',
                ])
            
            if bitrate:
                x265_params.append(f'bitrate={bitrate}')
            else:
                x265_params.append(f'crf={crf}')
            
            stream.options = {'x265-params': ':'.join(x265_params)}
            
            for i in range(T):
                frame = av.VideoFrame.from_ndarray(arr[i], format='rgb24')
                for pkt in stream.encode(frame):
                    container.mux(pkt)
            
            for pkt in stream.encode():
                container.mux(pkt)
        
        compressed = out.getvalue()
        
        if verbose:
            mode = "Real-time intra-only" if intra_only else "Real-time inter"
            print(f"[H.265-RT] 模式: {mode} | CRF: {crf}")
            print(f"[H.265-RT] 配置: ultrafast preset (实时流媒体标准)")
            print(f"[H.265-RT] 压缩后: {len(compressed)/1024:.2f} KB")
            print(f"[H.265-RT] 说明: 模拟视频会议/云游戏等低延迟场景")
        
        return compressed
def _compress_to_h264_forexpert(
    video_tensor, 
    fps=30, 
    crf=45,          
    intra_only=False,
    verbose=False
):
    if video_tensor.dim() != 4 or video_tensor.shape[1] != 3:
        raise ValueError("Input tensor must be (T, C, H, W) with C=3.")
    
    T, C, H, W = video_tensor.shape
    if H == 0 or W == 0: return b''
    
    pad_w = W % 2
    pad_h = H % 2
    if pad_w or pad_h:
        video_tensor = F.pad(video_tensor, (0, pad_w, 0, pad_h))
        H, W = video_tensor.shape[2:]
    
    arr = (video_tensor.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    out = io.BytesIO()
    
    with av.open(out, mode='w', format='mp4') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = 'yuv420p'
        
        # 📉 H.264 低 PSNR 配置
        x264_params = [
            'preset=ultrafast',
            f'crf={crf}',            # 高 CRF
            
            # --- 增加误差的操作 ---
            # 开启去块滤波，甚至可以加强它 (alpha:beta)，让画面更糊
            # 默认是开启的，这里显式写出。
            'deblock=1:0:0',         
            
            # 禁用 Psy-RD。
            # Psy-RD 会尝试保留噪点和纹理以欺骗人眼（提高视觉质量但可能降低 PSNR）。
            # 禁用它 (0) 会让编码器只追求压缩率，导致纹理被抹平，从而增大与原图的误差。
            'psy-rd=0.0:0.0', 
            
            # --- 糟糕的预测 ---
            'me=dia',                # 钻石搜索
            'subme=0',               # 无子像素精细度
            'merange=4',             # 极小的搜索范围
            'no-chroma-me=1',        # 放弃色度运动估计
            
            # --- 禁用复杂算法 ---
            'no-cabac=1',            # 使用 CAVLC，效率低
            'trellis=0',             # 禁用网格量化
            'partitions=none',       # 禁用分区，强行用大块
        ]

        if intra_only:
            x264_params.extend(['keyint=1', 'min-keyint=1', 'scenecut=0'])
        else:
            x264_params.extend(['keyint=250', 'scenecut=0'])

        stream.options = {'x264-params': ':'.join(x264_params)}
        
        for i in range(T):
            frame = av.VideoFrame.from_ndarray(arr[i], format='rgb24')
            for pkt in stream.encode(frame):
                container.mux(pkt)
        
        for pkt in stream.encode():
            container.mux(pkt)
            
    compressed = out.getvalue()
    
    if verbose:
        print(f"[H.264-LOW-PSNR] CRF: {crf} | Deblock: On | Psy: Off")

    return compressed