import cv2
import numpy as np
import torch
from tqdm import tqdm
from decord import VideoReader, cpu
from rich import print

def read_video_to_array(video_path, max_frames=None):
    """
    优化的视频读取函数
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        print(f"Error initializing decord VideoReader: {e}")
        return read_video_to_array_cv2(video_path, max_frames)
        
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    # 优化：直接从第一帧获取尺寸，避免多次访问
    first_frame = vr[0]
    height, width = first_frame.shape[:2]
    
    print(f"Video Info: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    frames_to_read = min(max_frames, total_frames) if max_frames else total_frames
    
    if frames_to_read == 0:
        print("Warning: Number of frames to read is 0.")
        return np.empty((0, 3, height, width), dtype=np.uint8), (fps, width, height)

    # 批量读取并转置（decord已经很快了）
    video_array_hwc = vr.get_batch(list(range(frames_to_read))).asnumpy()
    video_array = np.transpose(video_array_hwc, (0, 3, 1, 2))
    
    print(f"Successfully read video. Shape: {video_array.shape}")
    return video_array, (fps, width, height)


def save_tensor_as_video(tensor, output_path, fps=30, fourcc='mp4v'):
    """
    优化版本：批量处理所有帧，减少循环开销
    """
    if tensor is None or tensor.shape[0] == 0:
        print("Warning: Input tensor is empty, cannot save video.")
        return
    
    # 转换为numpy数组
    if torch.is_tensor(tensor):
        video_array = tensor.cpu().numpy()
    else:
        video_array = tensor
    
    # 归一化处理
    if video_array.max() <= 1.0 and video_array.dtype != np.uint8:
        video_array = (video_array * 255).astype(np.uint8)
    
    t, c, h, w = video_array.shape
    
    # 批量转置：(t, c, h, w) -> (t, h, w, c)
    video_array_hwc = np.transpose(video_array, (0, 2, 3, 1))
    
    # 批量颜色转换：RGB -> BGR
    if c == 3:
        video_array_bgr = video_array_hwc[..., ::-1].copy()  # 更快的颜色通道反转
    else:
        video_array_bgr = video_array_hwc
    
    # 写入视频
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(output_path, fourcc_code, float(fps), (w, h))
    
    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}")
    
    # 批量写入（仍需循环，但减少了每次的计算）
    for i in tqdm(range(t), desc="Saving video", disable=t<100):
        out.write(video_array_bgr[i])
    
    out.release()
    print(f"✅ Video saved to {output_path}")

