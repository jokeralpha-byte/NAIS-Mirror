import numpy as np
import math
import os
import numpy as np
import math
import os
# ========== C++ 加速版本导入 ==========
try:
    from module.utils.viewport_ops_cpp  import (
        real_gen_viewport_chunks as _real_gen_viewport_chunks_cpp,
        paste_viewport_to_video as _paste_viewport_to_video_cpp,
        get_num_threads,
        set_num_threads
    )
    _CPP_AVAILABLE = True
    print(f"✅ [chunk_processor] C++ accelerated version loaded (OpenMP threads: {get_num_threads()})")
except ImportError as e:
    _CPP_AVAILABLE = False
    print(f"⚠️  [chunk_processor] C++ version not available: {e}")
    print("    Falling back to pure Python implementation")

def _real_gen_viewport_chunks_python(video_frames, total_frames, angel, indx, max_size=(3840, 2160), chunk_length=10, chunk_size=(1200, 1200)):
    # Generates the video chunk for transmission.
    # Format is (t, c, h, w)
    start_index = indx * chunk_length
    end_index = start_index + chunk_length

    if end_index > total_frames:
        total_video = video_frames[start_index:]
    else:
        total_video = video_frames[start_index:end_index]
    
    actual_chunk_length = total_video.shape[0]
    if actual_chunk_length == 0:
        return np.zeros((0, 3, chunk_size[1], chunk_size[0]), dtype=video_frames.dtype)


    chunk_w, chunk_h = chunk_size
    half_w, half_h = chunk_w // 2, chunk_h // 2

    yaw, pitch = angel
    yaw_min = half_w
    yaw_max = max_size[0] - half_w
    pitch_min = half_h
    pitch_max = max_size[1] - half_h

    center_x = int(max_size[0] * (yaw + 180) / 360)
    center_y = int(max_size[1] * (90 - pitch) / 180)
    
    output_frame = np.zeros((actual_chunk_length, 3, chunk_h, chunk_w), dtype=total_video.dtype)

    # Case 1: Center is within the normal range
    if yaw_min <= center_x <= yaw_max and pitch_min <= center_y <= pitch_max:
        output_frame[:, :, :, :] = total_video[:, :, center_y - half_h:center_y + half_h, center_x - half_w:center_x + half_w]
    
    # Case 2: Top-left corner wrap-around
    elif center_x < yaw_min and center_y < pitch_min:
        wrap_w = yaw_min - center_x
        wrap_h = pitch_min - center_y
        
        # Top-left of output -> Bottom-right of source
        output_frame[:, :, :wrap_h, :wrap_w] = total_video[:, :, max_size[1] - wrap_h:, max_size[0] - wrap_w:]
        # Top-right of output -> Bottom-left of source
        output_frame[:, :, :wrap_h, wrap_w:] = total_video[:, :, max_size[1] - wrap_h:, :center_x + half_w]
        # Bottom-left of output -> Top-right of source
        output_frame[:, :, wrap_h:, :wrap_w] = total_video[:, :, :center_y + half_h, max_size[0] - wrap_w:]
        # Bottom-right of output -> Top-left of source
        output_frame[:, :, wrap_h:, wrap_w:] = total_video[:, :, :center_y + half_h, :center_x + half_w]
    
    # Case 3: Top-right corner wrap-around
    elif center_x > yaw_max and center_y < pitch_min:
        wrap_w = center_x - yaw_max
        wrap_h = pitch_min - center_y
        width_before_wrap = chunk_w - wrap_w

        # Top-left of output -> Bottom-right of source
        output_frame[:, :, :wrap_h, :width_before_wrap] = total_video[:, :, max_size[1] - wrap_h:, center_x - half_w:]
        # Top-right of output -> Bottom-left of source
        output_frame[:, :, :wrap_h, width_before_wrap:] = total_video[:, :, max_size[1] - wrap_h:, :wrap_w]
        # Bottom-left of output -> Top-right of source
        output_frame[:, :, wrap_h:, :width_before_wrap] = total_video[:, :, :center_y + half_h, center_x - half_w:]
        # Bottom-right of output -> Top-left of source
        output_frame[:, :, wrap_h:, width_before_wrap:] = total_video[:, :, :center_y + half_h, :wrap_w]

    # Case 4: Bottom-left corner wrap-around
    elif center_x < yaw_min and center_y > pitch_max:
        wrap_w = yaw_min - center_x
        wrap_h = center_y - pitch_max
        height_before_wrap = chunk_h - wrap_h
        
        # Top-left of output -> Bottom-right of source
        output_frame[:, :, :height_before_wrap, :wrap_w] = total_video[:, :, center_y - half_h:, max_size[0] - wrap_w:]
        # Top-right of output -> Bottom-left of source
        output_frame[:, :, :height_before_wrap, wrap_w:] = total_video[:, :, center_y - half_h:, :center_x + half_w]
        # Bottom-left of output -> Top-right of source
        output_frame[:, :, height_before_wrap:, :wrap_w] = total_video[:, :, :wrap_h, max_size[0] - wrap_w:]
        # Bottom-right of output -> Top-left of source
        output_frame[:, :, height_before_wrap:, wrap_w:] = total_video[:, :, :wrap_h, :center_x + half_w]

    # Case 5: Bottom-right corner wrap-around
    elif center_x > yaw_max and center_y > pitch_max:
        wrap_w = center_x - yaw_max
        wrap_h = center_y - pitch_max
        width_before_wrap = chunk_w - wrap_w
        height_before_wrap = chunk_h - wrap_h
        
        # Top-left of output -> Bottom-right of source
        output_frame[:, :, :height_before_wrap, :width_before_wrap] = total_video[:, :, center_y - half_h:, center_x - half_w:]
        # Top-right of output -> Bottom-left of source
        output_frame[:, :, :height_before_wrap, width_before_wrap:] = total_video[:, :, center_y - half_h:, :wrap_w]
        # Bottom-left of output -> Top-right of source
        output_frame[:, :, height_before_wrap:, :width_before_wrap] = total_video[:, :, :wrap_h, center_x - half_w:]
        # Bottom-right of output -> Top-left of source
        output_frame[:, :, height_before_wrap:, width_before_wrap:] = total_video[:, :, :wrap_h, :wrap_w]

    # Case 6: Left boundary wrap-around
    elif center_x < yaw_min:
        wrap_w = yaw_min - center_x
        output_frame[:, :, :, :wrap_w] = total_video[:, :, center_y - half_h:center_y + half_h, max_size[0] - wrap_w:]
        output_frame[:, :, :, wrap_w:] = total_video[:, :, center_y - half_h:center_y + half_h, :center_x + half_w]

    # Case 7: Right boundary wrap-around
    elif center_x > yaw_max:
        # CORRECTED LOGIC
        wrap_w = center_x - yaw_max
        width_before_wrap = chunk_w - wrap_w
        output_frame[:, :, :, :width_before_wrap] = total_video[:, :, center_y - half_h:center_y + half_h, center_x - half_w:]
        output_frame[:, :, :, width_before_wrap:] = total_video[:, :, center_y - half_h:center_y + half_h, :wrap_w]

    # Case 8: Top boundary wrap-around
    elif center_y < pitch_min:
        wrap_h = pitch_min - center_y
        output_frame[:, :, :wrap_h, :] = total_video[:, :, max_size[1] - wrap_h:, center_x - half_w:center_x + half_w]
        output_frame[:, :, wrap_h:, :] = total_video[:, :, :center_y + half_h, center_x - half_w:center_x + half_w]

    # Case 9: Bottom boundary wrap-around (The one that caused the error)
    else:  # center_y > pitch_max
        # CORRECTED LOGIC
        wrap_h = center_y - pitch_max
        height_before_wrap = chunk_h - wrap_h
        output_frame[:, :, :height_before_wrap, :] = total_video[:, :, center_y - half_h:, center_x - half_w:center_x + half_w]
        output_frame[:, :, height_before_wrap:, :] = total_video[:, :, :wrap_h, center_x - half_w:center_x + half_w]

    return output_frame

def _paste_viewport_to_video_python(video_frames, viewport_chunk, angel, indx, max_size=(3840, 2160), chunk_length=10, chunk_size=(1200, 1200), border_thickness=10):
    """
    Pastes the generated viewport chunk with an optional inner black border
    back to the original video position.
    
    Args:
        ... (original arguments) ...
        border_thickness (int): The thickness of the black border to add inside the chunk.
                                Set to 0 for no border.
    """
    # Pastes the generated viewport chunk back to the original video position.
    # Note: Corrected all wrap-around logic to match the generator function.
    if len(video_frames) <= indx * chunk_length + chunk_length:
        target_video_slice = video_frames[indx * chunk_length:]
    else:
        target_video_slice = video_frames[indx * chunk_length:indx * chunk_length + chunk_length]
    
    actual_chunk_length = len(target_video_slice)
    # 使用 .copy() 以确保我们不会修改原始的 video_frames 数据
    output_video = target_video_slice.copy()
    
    chunk_w, chunk_h = chunk_size
    half_w, half_h = chunk_w // 2, chunk_h // 2
    
    yaw, pitch = angel
    yaw_min = half_w
    yaw_max = max_size[0] - half_w
    pitch_min = half_h
    pitch_max = max_size[1] - half_h
    
    center_x = int(max_size[0] * (yaw + 180) / 360)
    center_y = int(max_size[1] * (90 - pitch) / 180)
    
    # Slicing the viewport_chunk to match the actual length of the video slice
    viewport_chunk_sliced = viewport_chunk[:actual_chunk_length]
    # ==================== 新增代码：为 chunk 添加黑边 ====================
    if border_thickness > 0 and viewport_chunk_sliced.size > 0:
        # 复制一份以避免修改原始的 viewport_chunk 数据
        viewport_chunk_sliced = viewport_chunk_sliced.copy()
        
        # 获取 chunk 的实际高和宽 (shape: [frames, channels, height, width])
        h, w = viewport_chunk_sliced.shape[2], viewport_chunk_sliced.shape[3]
        
        # 将顶部和底部的边框像素设置为0（黑色）
        viewport_chunk_sliced[:, :, :border_thickness, :] = 0
        viewport_chunk_sliced[:, :, h - border_thickness:, :] = 0
        
        # 将左侧和右侧的边框像素设置为0（黑色）
        viewport_chunk_sliced[:, :, :, :border_thickness] = 0
        viewport_chunk_sliced[:, :, :, w - border_thickness:] = 0
    # =================================================================
    # Case 1: Center is within the normal range (后续所有粘贴逻辑保持不变)
    if yaw_min <= center_x <= yaw_max and pitch_min <= center_y <= pitch_max:
        output_video[:, :, center_y - half_h:center_y + half_h, center_x - half_w:center_x + half_w] = viewport_chunk_sliced
    
    # Case 2: Top-left corner wrap-around
    elif center_x < yaw_min and center_y < pitch_min:
        wrap_w = yaw_min - center_x
        wrap_h = pitch_min - center_y
        output_video[:, :, max_size[1] - wrap_h:, max_size[0] - wrap_w:] = viewport_chunk_sliced[:, :, :wrap_h, :wrap_w]
        output_video[:, :, max_size[1] - wrap_h:, :center_x + half_w] = viewport_chunk_sliced[:, :, :wrap_h, wrap_w:]
        output_video[:, :, :center_y + half_h, max_size[0] - wrap_w:] = viewport_chunk_sliced[:, :, wrap_h:, :wrap_w]
        output_video[:, :, :center_y + half_h, :center_x + half_w] = viewport_chunk_sliced[:, :, wrap_h:, wrap_w:]
    # ... (其他所有 elif 和 else 分支代码保持完全不变) ...
    # Case 3: Top-right corner wrap-around
    elif center_x > yaw_max and center_y < pitch_min:
        wrap_w = center_x - yaw_max
        wrap_h = pitch_min - center_y
        width_before_wrap = chunk_w - wrap_w
        output_video[:, :, max_size[1] - wrap_h:, center_x - half_w:] = viewport_chunk_sliced[:, :, :wrap_h, :width_before_wrap]
        output_video[:, :, max_size[1] - wrap_h:, :wrap_w] = viewport_chunk_sliced[:, :, :wrap_h, width_before_wrap:]
        output_video[:, :, :center_y + half_h, center_x - half_w:] = viewport_chunk_sliced[:, :, wrap_h:, :width_before_wrap]
        output_video[:, :, :center_y + half_h, :wrap_w] = viewport_chunk_sliced[:, :, wrap_h:, width_before_wrap:]
    # Case 4: Bottom-left corner wrap-around
    elif center_x < yaw_min and center_y > pitch_max:
        wrap_w = yaw_min - center_x
        wrap_h = center_y - pitch_max
        height_before_wrap = chunk_h - wrap_h
        output_video[:, :, center_y - half_h:, max_size[0] - wrap_w:] = viewport_chunk_sliced[:, :, :height_before_wrap, :wrap_w]
        output_video[:, :, center_y - half_h:, :center_x + half_w] = viewport_chunk_sliced[:, :, :height_before_wrap, wrap_w:]
        output_video[:, :, :wrap_h, max_size[0] - wrap_w:] = viewport_chunk_sliced[:, :, height_before_wrap:, :wrap_w]
        output_video[:, :, :wrap_h, :center_x + half_w] = viewport_chunk_sliced[:, :, height_before_wrap:, wrap_w:]
    # Case 5: Bottom-right corner wrap-around
    elif center_x > yaw_max and center_y > pitch_max:
        wrap_w = center_x - yaw_max
        wrap_h = center_y - pitch_max
        width_before_wrap = chunk_w - wrap_w
        height_before_wrap = chunk_h - wrap_h
        output_video[:, :, center_y - half_h:, center_x - half_w:] = viewport_chunk_sliced[:, :, :height_before_wrap, :width_before_wrap]
        output_video[:, :, center_y - half_h:, :wrap_w] = viewport_chunk_sliced[:, :, :height_before_wrap, width_before_wrap:]
        output_video[:, :, :wrap_h, center_x - half_w:] = viewport_chunk_sliced[:, :, height_before_wrap:, :width_before_wrap]
        output_video[:, :, :wrap_h, :wrap_w] = viewport_chunk_sliced[:, :, height_before_wrap:, width_before_wrap:]
    # Case 6: Left boundary wrap-around
    elif center_x < yaw_min:
        wrap_w = yaw_min - center_x
        output_video[:, :, center_y - half_h:center_y + half_h, max_size[0] - wrap_w:] = viewport_chunk_sliced[:, :, :, :wrap_w]
        output_video[:, :, center_y - half_h:center_y + half_h, :center_x + half_w] = viewport_chunk_sliced[:, :, :, wrap_w:]
    # Case 7: Right boundary wrap-around
    elif center_x > yaw_max:
        wrap_w = center_x - yaw_max
        width_before_wrap = chunk_w - wrap_w
        output_video[:, :, center_y - half_h:center_y + half_h, center_x - half_w:] = viewport_chunk_sliced[:, :, :, :width_before_wrap]
        output_video[:, :, center_y - half_h:center_y + half_h, :wrap_w] = viewport_chunk_sliced[:, :, :, width_before_wrap:]
    # Case 8: Top boundary wrap-around
    elif center_y < pitch_min:
        wrap_h = pitch_min - center_y
        output_video[:, :, max_size[1] - wrap_h:, center_x - half_w:center_x + half_w] = viewport_chunk_sliced[:, :, :wrap_h, :]
        output_video[:, :, :center_y + half_h, center_x - half_w:center_x + half_w] = viewport_chunk_sliced[:, :, wrap_h:, :]
    # Case 9: Bottom boundary wrap-around
    else:  # center_y > pitch_max
        wrap_h = center_y - pitch_max
        height_before_wrap = chunk_h - wrap_h
        output_video[:, :, center_y - half_h:, center_x - half_w:center_x + half_w] = viewport_chunk_sliced[:, :, :height_before_wrap, :]
        output_video[:, :, :wrap_h, center_x - half_w:center_x + half_w] = viewport_chunk_sliced[:, :, height_before_wrap:, :]
    return output_video


def generate_chunks_pos(total_frames,if_gen=True,chunk_length=10,output_angel_dir=None,max_yaw=45,max_pitch=30):
    '''
    total_frames: 总帧数
    if_gen: 是否生成路径
    chunk_length: 每个chunk的长度
    output_angel_dir: 输出到文件中的位置
    生成每个chunk实时的位置，后期将使用实时读取来进行模拟
    '''
    if if_gen:

        point_num=0
        if total_frames <= 0:
            gen_angel=[]
        else:
            point_num=math.ceil(total_frames/chunk_length)
            last_yaw_angel=0
            last_pitch_angel=0

            if os.path.exists(output_angel_dir):
                #如果存在就直接删除新建一个
                os.remove(output_angel_dir)
            with open(output_angel_dir, 'w') as f:
                f.close()
            for i in range(point_num):
                while True:
                    waypoint_yaws = np.random.uniform(-170, 170, 1)
                    waypoint_pitches = np.random.uniform(-60, 60, 1)
                    if abs(waypoint_yaws[0]-last_yaw_angel) <= max_yaw and abs(waypoint_pitches[0]-last_pitch_angel) <= max_pitch:
                        last_yaw_angel=waypoint_yaws[0]
                        last_pitch_angel=waypoint_pitches[0]
                        break


                
                with open(output_angel_dir, 'a') as f:
                    f.write(f"{waypoint_yaws[0]:.6f},{waypoint_pitches[0]:.6f}\n")
            print(f'路径保存在{output_angel_dir}')

def real_gen_viewport_chunks(video_frames, total_frames, angel, indx, 
                            max_size=(3840, 2160), chunk_length=10, 
                            chunk_size=(1200, 1200), force_python=False):
    """自动选择C++或Python版本"""
    if _CPP_AVAILABLE and not force_python:
        print("Using C++ version for generating viewport chunks.")
        return _real_gen_viewport_chunks_cpp(
            video_frames, total_frames, angel, indx, 
            max_size, chunk_length, chunk_size
        )
    else:
        print("Using Python version for generating viewport chunks.")
        return _real_gen_viewport_chunks_python(
            video_frames, total_frames, angel, indx, 
            max_size, chunk_length, chunk_size
        )
def paste_viewport_to_video(video_frames, viewport_chunk, angel, indx, 
                           max_size=(3840, 2160), chunk_length=10, 
                           chunk_size=(1200, 1200), border_thickness=10,
                           force_python=False):
    """自动选择C++或Python版本"""
    if _CPP_AVAILABLE and not force_python:
        print("Using C++ version for pasting viewport to video.")
        return _paste_viewport_to_video_cpp(
            video_frames, viewport_chunk, angel, indx,
            max_size, chunk_length, chunk_size, border_thickness
        )
    else:
        print("Using Python version for pasting viewport to video.")
        return _paste_viewport_to_video_python(
            video_frames, viewport_chunk, angel, indx,
            max_size, chunk_length, chunk_size, border_thickness
        )
# ========== OpenMP 线程控制 ==========
def set_openmp_threads(n):
    """设置OpenMP线程数"""
    if _CPP_AVAILABLE:
        set_num_threads(n)
        print(f"✅ OpenMP threads set to {n}")
    else:
        print("⚠️  C++ module not available, cannot set threads")
def get_openmp_threads():
    """获取当前OpenMP线程数"""
    if _CPP_AVAILABLE:
        return get_num_threads()
    else:
        return 1