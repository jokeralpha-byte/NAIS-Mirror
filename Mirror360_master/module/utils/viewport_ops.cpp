#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>      // For std::memcpy, std::memset
#include <stdexcept>    // For std::runtime_error
#include <sstream>      // For std::ostringstream
#include <algorithm>    // For std::min, std::max

// 检查 OpenMP 是否可用
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ============================================================================
//                          !! 安全检查宏 !!
// ============================================================================
#define SAFE_CHECK(condition, message) \
    if (!(condition)) { \
        std::ostringstream oss; \
        oss << "C++ Safety Check Failed at " << __FILE__ << ":" << __LINE__ \
            << " - " << message; \
        throw std::runtime_error(oss.str()); \
    }

// ============================================================================
//
//                     核心函数 1 (已修复): 
//         安全复制4D视频区域 (Stride-Aware)
//
// ============================================================================
/**
 * @brief (Stride-Aware) 安全地从源 4D 数组复制一个区域到目标 4D 数组。
 * 目标(dst)被假定为 C-Contiguous。
 * 源(src)使用传入的 strides 进行访问。
 */
inline void safe_copy_region_4d(
    uint8_t* dst_ptr, const uint8_t* src_ptr,
    int n_frames, int n_channels,
    int dst_h_start, int dst_w_start,
    int src_h_start, int src_w_start,
    int copy_h, int copy_w,
    int dst_h_total, int dst_w_total, // 用于 Dst 步长
    int src_h_total, int src_w_total, // 用于安全检查
    const py::ssize_t* src_strides   // 源步长 (来自 buf.strides())
) {
    // ------------------- 安全检查 -------------------
    SAFE_CHECK(dst_ptr != nullptr && src_ptr != nullptr, "Null pointer detected");
    SAFE_CHECK(n_frames > 0 && n_channels > 0, "Invalid frame/channel count");
    SAFE_CHECK(copy_h > 0 && copy_w > 0, "Invalid copy dimensions");
    SAFE_CHECK(dst_h_start >= 0 && dst_w_start >= 0, "Negative destination start");
    SAFE_CHECK(src_h_start >= 0 && src_w_start >= 0, "Negative source start");
    SAFE_CHECK(src_h_start + copy_h <= src_h_total, "Source height read overflow");
    SAFE_CHECK(src_w_start + copy_w <= src_w_total, "Source width read overflow");
    SAFE_CHECK(dst_h_start + copy_h <= dst_h_total, "Destination height write overflow");
    SAFE_CHECK(dst_w_start + copy_w <= dst_w_total, "Destination width write overflow");
    // ------------------------------------------------

    // 目标(Dst)步长 (C-contiguous, 单位: bytes)
    const size_t dst_stride_w_bytes = sizeof(uint8_t);
    const size_t dst_stride_h_bytes = static_cast<size_t>(dst_w_total) * dst_stride_w_bytes;
    const size_t dst_stride_c_bytes = static_cast<size_t>(dst_h_total) * dst_stride_h_bytes;
    const size_t dst_stride_t_bytes = static_cast<size_t>(n_channels) * dst_stride_c_bytes;

    // 源(Src)步长 (来自 numpy, 单位: bytes)
    const py::ssize_t src_stride_t_bytes = src_strides[0];
    const py::ssize_t src_stride_c_bytes = src_strides[1];
    const py::ssize_t src_stride_h_bytes = src_strides[2];
    const py::ssize_t src_stride_w_bytes = src_strides[3];

    const size_t row_copy_bytes = copy_w * sizeof(uint8_t);
    const bool src_row_is_contiguous = (src_stride_w_bytes == sizeof(uint8_t));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int t = 0; t < n_frames; ++t) {
        for (int c = 0; c < n_channels; ++c) {
            // (t, c) 的基地址
            const uint8_t* src_base_tc = src_ptr + t * src_stride_t_bytes + c * src_stride_c_bytes;
            uint8_t* dst_base_tc = dst_ptr + t * dst_stride_t_bytes + c * dst_stride_c_bytes;
            
            // 复制区域的起始行地址
            const uint8_t* src_row_start_ptr = src_base_tc + src_h_start * src_stride_h_bytes + src_w_start * src_stride_w_bytes;
            uint8_t* dst_row_start_ptr = dst_base_tc + dst_h_start * dst_stride_h_bytes + dst_w_start * dst_stride_w_bytes;

            if (src_row_is_contiguous) {
                // 优化：如果源数据在行(w)上是连续的，逐行使用 memcpy
                for (int h = 0; h < copy_h; ++h) {
                    const uint8_t* src_row = src_row_start_ptr + h * src_stride_h_bytes;
                    uint8_t* dst_row = dst_row_start_ptr + h * dst_stride_h_bytes;
                    std::memcpy(dst_row, src_row, row_copy_bytes);
                }
            } else {
                // 回退：如果源数据在行(w)上不连续，逐像素复制
                for (int h = 0; h < copy_h; ++h) {
                    const uint8_t* src_pixel = src_row_start_ptr + h * src_stride_h_bytes;
                    uint8_t* dst_pixel = dst_row_start_ptr + h * dst_stride_h_bytes;
                    for (int w = 0; w < copy_w; ++w) {
                        *dst_pixel = *src_pixel;
                        dst_pixel++; // Dst 是连续的
                        src_pixel += src_stride_w_bytes; // Src 按步长跳跃
                    }
                }
            }
        }
    }
}

// ============================================================================
//
//                     核心函数 2 (已修复): 
//         安全复制并添加黑边 (Stride-Aware)
//
// ============================================================================
/**
 * @brief (Stride-Aware) 类似于 safe_copy_region_4d，但在写入时动态添加黑边。
 * 目标(dst)被假定为 C-Contiguous。
 * 源(src)使用传入的 strides 进行访问。
 */
inline void safe_copy_with_border(
    uint8_t* dst_ptr, const uint8_t* src_ptr,
    int n_frames, int n_channels,
    int dst_h_start, int dst_w_start,
    int src_h_start, int src_w_start,
    int copy_h, int copy_w,
    int dst_h_total, int dst_w_total, // Dst 是 C-contiguous
    int src_h_total, int src_w_total, // 用于安全检查
    const py::ssize_t* src_strides,  // Src has strides
    int border_thickness
) {
    // ------------------- 安全检查 -------------------
    SAFE_CHECK(dst_ptr != nullptr && src_ptr != nullptr, "Null pointer detected");
    SAFE_CHECK(n_frames > 0 && n_channels > 0, "Invalid frame/channel count");
    SAFE_CHECK(copy_h > 0 && copy_w > 0, "Invalid copy dimensions");
    SAFE_CHECK(dst_h_start >= 0 && dst_w_start >= 0, "Negative destination start");
    SAFE_CHECK(src_h_start >= 0 && src_w_start >= 0, "Negative source start");
    SAFE_CHECK(src_h_start + copy_h <= src_h_total, "Source height read overflow");
    SAFE_CHECK(src_w_start + copy_w <= src_w_total, "Source width read overflow");
    SAFE_CHECK(dst_h_start + copy_h <= dst_h_total, "Destination height write overflow");
    SAFE_CHECK(dst_w_start + copy_w <= dst_w_total, "Destination width write overflow");
    SAFE_CHECK(border_thickness >= 0, "Negative border thickness");
    // ------------------------------------------------

    // Dst 步长 (C-contiguous, 单位: bytes)
    const size_t dst_stride_w_bytes = sizeof(uint8_t);
    const size_t dst_stride_h_bytes = static_cast<size_t>(dst_w_total) * dst_stride_w_bytes;
    const size_t dst_stride_c_bytes = static_cast<size_t>(dst_h_total) * dst_stride_h_bytes;
    const size_t dst_stride_t_bytes = static_cast<size_t>(n_channels) * dst_stride_c_bytes;
    
    // Src 步长 (来自 numpy, 单位: bytes)
    const py::ssize_t src_stride_t_bytes = src_strides[0];
    const py::ssize_t src_stride_c_bytes = src_strides[1];
    const py::ssize_t src_stride_h_bytes = src_strides[2];
    const py::ssize_t src_stride_w_bytes = src_strides[3];

    const bool no_border = (border_thickness == 0);
    const bool src_row_is_contiguous = (src_stride_w_bytes == sizeof(uint8_t));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int t = 0; t < n_frames; ++t) {
        for (int c = 0; c < n_channels; ++c) {
            // (t, c) 的基地址
            const uint8_t* src_base_tc = src_ptr + t * src_stride_t_bytes + c * src_stride_c_bytes;
            uint8_t* dst_base_tc = dst_ptr + t * dst_stride_t_bytes + c * dst_stride_c_bytes;
            
            for (int h = 0; h < copy_h; ++h) {
                // (h, w) 是相对于 *copy_h* 和 *copy_w* 的局部坐标
                int src_h_abs = src_h_start + h;
                int dst_h_abs = dst_h_start + h;

                // 目标行指针
                uint8_t* dst_row = dst_base_tc + dst_h_abs * dst_stride_h_bytes + dst_w_start * dst_stride_w_bytes;
                
                // 检查是否为顶部或底部黑边
                if (!no_border && (h < border_thickness || h >= copy_h - border_thickness)) {
                    std::memset(dst_row, 0, copy_w * sizeof(uint8_t));
                    continue; // 处理下一行
                }

                // --- 处理非顶部/底部的行 ---
                // 源行起始指针
                const uint8_t* src_row_start_ptr = src_base_tc + src_h_abs * src_stride_h_bytes + src_w_start * src_stride_w_bytes;
                
                // 1. 绘制左黑边
                int left_border_px = no_border ? 0 : border_thickness;
                if (left_border_px > 0) {
                    std::memset(dst_row, 0, left_border_px * sizeof(uint8_t));
                }
                
                // 2. 复制中间内容
                int right_border_start_px = no_border ? copy_w : std::max(0, copy_w - border_thickness);
                int middle_width_px = right_border_start_px - left_border_px;
                
                if (middle_width_px > 0) {
                    uint8_t* dst_middle = dst_row + left_border_px;
                    const uint8_t* src_middle = src_row_start_ptr + left_border_px * src_stride_w_bytes;
                    
                    if (src_row_is_contiguous) {
                        // 优化：源行连续，使用 memcpy
                        std::memcpy(dst_middle, src_middle, middle_width_px * sizeof(uint8_t));
                    } else {
                        // 回退：源行不连续，逐像素
                        for(int w = 0; w < middle_width_px; ++w) {
                            dst_middle[w] = *(src_middle + w * src_stride_w_bytes);
                        }
                    }
                }
                
                // 3. 绘制右黑边
                int right_border_width_px = copy_w - right_border_start_px;
                if (right_border_width_px > 0) {
                    std::memset(dst_row + right_border_start_px, 0, right_border_width_px * sizeof(uint8_t));
                }
            }
        }
    }
}


// ============================================================================
//
//                 PYTHON 绑定函数 1 (已修复): 
//              _real_gen_viewport_chunks_cpp
//
// ============================================================================
py::array_t<uint8_t> real_gen_viewport_chunks_cpp(
    py::array_t<uint8_t> video_frames,
    int total_frames,
    py::tuple angel,
    int indx,
    py::tuple max_size,
    int chunk_length,
    py::tuple chunk_size
) {
    // 1. 获取输入 Numpy 数组的缓冲区信息
    auto buf = video_frames.request();
    SAFE_CHECK(buf.ndim == 4, "video_frames must be 4D (t, c, h, w)");
    SAFE_CHECK(buf.shape[1] == 3, "video_frames must have 3 channels");
    SAFE_CHECK(buf.itemsize == sizeof(uint8_t), "video_frames must be uint8");
    
    // 2. 解析参数
    const int max_w = max_size[0].cast<int>();
    const int max_h = max_size[1].cast<int>();
    const int chunk_w = chunk_size[0].cast<int>();
    const int chunk_h = chunk_size[1].cast<int>();
    
    SAFE_CHECK(buf.shape[2] == max_h, "Video height mismatch max_size[1]");
    SAFE_CHECK(buf.shape[3] == max_w, "Video width mismatch max_size[0]");

    // 3. 计算帧范围
    const int start_index = indx * chunk_length;
    const int end_index = std::min(start_index + chunk_length, total_frames);
    const int actual_chunk_length = end_index - start_index;

    if (actual_chunk_length <= 0) {
        return py::array_t<uint8_t>({0, 3, chunk_h, chunk_w});
    }

    // 4. 计算坐标 (与 python 逻辑一致)
    const double yaw = angel[0].cast<double>();
    const double pitch = angel[1].cast<double>();
    const int half_w = chunk_w / 2, half_h = chunk_h / 2;
    const int yaw_min = half_w, yaw_max = max_w - half_w;
    const int pitch_min = half_h, pitch_max = max_h - half_h;
    const int center_x = static_cast<int>(max_w * (yaw + 180.0) / 360.0);
    const int center_y = static_cast<int>(max_h * (90.0 - pitch) / 180.0);
    
    // 5. 创建输出数组 (C-contiguous)
    py::array_t<uint8_t> output({actual_chunk_length, 3, chunk_h, chunk_w});
    std::memset(output.mutable_data(), 0, output.nbytes());
    uint8_t* out_ptr = output.mutable_data();
    
    // 6. 获取源指针和步长 (FIXED)
    const uint8_t* video_ptr = static_cast<const uint8_t*>(buf.ptr);
    
    // ================== C++ 编译错误修复 ==================
    const py::ssize_t* src_strides = buf.strides.data();
    // ======================================================
    
    // 将输入指针偏移到正确的起始帧
    const uint8_t* in_ptr = video_ptr + start_index * src_strides[0];

    // 7. 9种环绕情况的逻辑 (FIXED: 传入 src_strides)
    try {
        // Case 1: 正常
        if (center_x >= yaw_min && center_x <= yaw_max && 
            center_y >= pitch_min && center_y <= pitch_max) {
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3,
                               0, 0, center_y - half_h, center_x - half_w,
                               chunk_h, chunk_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 2: 左上角
        else if (center_x < yaw_min && center_y < pitch_min) {
            int wrap_w = yaw_min - center_x; int wrap_h = pitch_min - center_y;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, max_h - wrap_h, max_w - wrap_w, wrap_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, wrap_w, max_h - wrap_h, 0, wrap_h, center_x + half_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, wrap_h, 0, 0, max_w - wrap_w, center_y + half_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, wrap_h, wrap_w, 0, 0, center_y + half_h, center_x + half_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 3: 右上角
        else if (center_x > yaw_max && center_y < pitch_min) {
            int wrap_w = center_x - yaw_max; int wrap_h = pitch_min - center_y; int width_before_wrap = chunk_w - wrap_w;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, max_h - wrap_h, center_x - half_w, wrap_h, width_before_wrap, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, width_before_wrap, max_h - wrap_h, 0, wrap_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, wrap_h, 0, 0, center_x - half_w, center_y + half_h, width_before_wrap, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, wrap_h, width_before_wrap, 0, 0, center_y + half_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 4: 左下角
        else if (center_x < yaw_min && center_y > pitch_max) {
            int wrap_w = yaw_min - center_x; int wrap_h = center_y - pitch_max; int height_before_wrap = chunk_h - wrap_h;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, center_y - half_h, max_w - wrap_w, height_before_wrap, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, wrap_w, center_y - half_h, 0, height_before_wrap, center_x + half_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, height_before_wrap, 0, 0, max_w - wrap_w, wrap_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, height_before_wrap, wrap_w, 0, 0, wrap_h, center_x + half_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 5: 右下角
        else if (center_x > yaw_max && center_y > pitch_max) {
            int wrap_w = center_x - yaw_max; int wrap_h = center_y - pitch_max; int width_before_wrap = chunk_w - wrap_w; int height_before_wrap = chunk_h - wrap_h;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, center_y - half_h, center_x - half_w, height_before_wrap, width_before_wrap, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, width_before_wrap, center_y - half_h, 0, height_before_wrap, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, height_before_wrap, 0, 0, center_x - half_w, wrap_h, width_before_wrap, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, height_before_wrap, width_before_wrap, 0, 0, wrap_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 6: 左边界
        else if (center_x < yaw_min) {
            int wrap_w = yaw_min - center_x;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, center_y - half_h, max_w - wrap_w, chunk_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, wrap_w, center_y - half_h, 0, chunk_h, center_x + half_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 7: 右边界
        else if (center_x > yaw_max) {
            int wrap_w = center_x - yaw_max; int width_before_wrap = chunk_w - wrap_w;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, center_y - half_h, center_x - half_w, chunk_h, width_before_wrap, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, width_before_wrap, center_y - half_h, 0, chunk_h, wrap_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 8: 上边界
        else if (center_y < pitch_min) {
            int wrap_h = pitch_min - center_y;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, max_h - wrap_h, center_x - half_w, wrap_h, chunk_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, wrap_h, 0, 0, center_x - half_w, center_y + half_h, chunk_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
        // Case 9: 下边界
        else { // center_y > pitch_max
            int wrap_h = center_y - pitch_max; int height_before_wrap = chunk_h - wrap_h;
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, 0, 0, center_y - half_h, center_x - half_w, height_before_wrap, chunk_w, chunk_h, chunk_w, max_h, max_w, src_strides);
            safe_copy_region_4d(out_ptr, in_ptr, actual_chunk_length, 3, height_before_wrap, 0, 0, center_x - half_w, wrap_h, chunk_w, chunk_h, chunk_w, max_h, max_w, src_strides);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in real_gen_viewport_chunks: ") + e.what());
    }
    
    return output;
}


// ============================================================================
//
//                 PYTHON 绑定函数 2 (已修复): 
//              _paste_viewport_to_video_cpp
//
// ============================================================================
py::array_t<uint8_t> paste_viewport_to_video_cpp(
    py::array_t<uint8_t> video_frames,
    py::array_t<uint8_t> viewport_chunk,
    py::tuple angel,
    int indx,
    py::tuple max_size,
    int chunk_length,
    py::tuple chunk_size,
    int border_thickness
) {
    // 1. 获取缓冲区
    auto video_buf = video_frames.request();
    auto chunk_buf = viewport_chunk.request();

    // 2. 安全检查
    SAFE_CHECK(video_buf.ndim == 4, "video_frames must be 4D");
    SAFE_CHECK(chunk_buf.ndim == 4, "viewport_chunk must be 4D");
    SAFE_CHECK(video_buf.shape[1] == 3, "video_frames must have 3 channels");
    SAFE_CHECK(chunk_buf.shape[1] == 3, "viewport_chunk must have 3 channels");
    SAFE_CHECK(video_buf.itemsize == sizeof(uint8_t), "video_frames must be uint8");
    SAFE_CHECK(chunk_buf.itemsize == sizeof(uint8_t), "viewport_chunk must be uint8");

    // 3. 解析参数
    const int max_w = max_size[0].cast<int>();
    const int max_h = max_size[1].cast<int>();
    const int chunk_w = chunk_size[0].cast<int>();
    const int chunk_h = chunk_size[1].cast<int>();
    
    SAFE_CHECK(video_buf.shape[2] == max_h, "Video height mismatch");
    SAFE_CHECK(video_buf.shape[3] == max_w, "Video width mismatch");
    SAFE_CHECK(chunk_buf.shape[2] == chunk_h, "Chunk height mismatch");
    SAFE_CHECK(chunk_buf.shape[3] == chunk_w, "Chunk width mismatch");

    // 4. 计算帧范围
    const int video_total_frames = static_cast<int>(video_buf.shape[0]);
    const int start_index = indx * chunk_length;
    const int end_index = std::min(start_index + chunk_length, video_total_frames);
    const int actual_chunk_length = end_index - start_index;

    if (actual_chunk_length <= 0) {
        return py::array_t<uint8_t>({0, 3, max_h, max_w});
    }
    SAFE_CHECK(static_cast<int>(chunk_buf.shape[0]) >= actual_chunk_length, "viewport_chunk has fewer frames than required");

    // 5. 创建输出数组 (C-contiguous)
    py::array_t<uint8_t> output({actual_chunk_length, 3, max_h, max_w});
    uint8_t* out_ptr = output.mutable_data();
    
    // 6. 获取源指针和步长 (FIXED)
    const uint8_t* video_ptr_start = static_cast<const uint8_t*>(video_buf.ptr);
    
    // ================== C++ 编译错误修复 ==================
    const py::ssize_t* video_strides = video_buf.strides.data();
    // ======================================================
    
    const uint8_t* video_ptr = video_ptr_start + start_index * video_strides[0];
    
    const uint8_t* chunk_ptr = static_cast<const uint8_t*>(chunk_buf.ptr);

    // ================== C++ 编译错误修复 ==================
    const py::ssize_t* chunk_strides = chunk_buf.strides.data();
    // ======================================================


    // 7. 将背景视频复制到新数组中 (FIXED: Stride-Aware)
    try {
        safe_copy_region_4d(
            out_ptr, video_ptr,
            actual_chunk_length, 3,
            0, 0, 0, 0,           // dst_start, src_start
            max_h, max_w,        // copy_h, copy_w
            max_h, max_w,        // dst_total_h, dst_total_w
            max_h, max_w,        // src_total_h, src_total_w
            video_strides
        );
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error copying background video: ") + e.what());
    }
    
    // 8. 计算坐标
    const double yaw = angel[0].cast<double>();
    const double pitch = angel[1].cast<double>();
    const int half_w = chunk_w / 2, half_h = chunk_h / 2;
    const int yaw_min = half_w, yaw_max = max_w - half_w;
    const int pitch_min = half_h, pitch_max = max_h - half_h;
    const int center_x = static_cast<int>(max_w * (yaw + 180.0) / 360.0);
    const int center_y = static_cast<int>(max_h * (90.0 - pitch) / 180.0);

    // 9. 9种环绕情况的逻辑 (粘贴) (FIXED: 传入 chunk_strides)
    try {
        // Case 1: 正常
        if (center_x >= yaw_min && center_x <= yaw_max && 
            center_y >= pitch_min && center_y <= pitch_max) {
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3,
                                 center_y - half_h, center_x - half_w, 0, 0,
                                 chunk_h, chunk_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 2: 左上角
        else if (center_x < yaw_min && center_y < pitch_min) {
            int wrap_w = yaw_min - center_x; int wrap_h = pitch_min - center_y;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, max_h - wrap_h, max_w - wrap_w, 0, 0, wrap_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, max_h - wrap_h, 0, 0, wrap_w, wrap_h, center_x + half_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, max_w - wrap_w, wrap_h, 0, center_y + half_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, 0, wrap_h, wrap_w, center_y + half_h, center_x + half_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 3: 右上角
        else if (center_x > yaw_max && center_y < pitch_min) {
            int wrap_w = center_x - yaw_max; int wrap_h = pitch_min - center_y; int width_before_wrap = chunk_w - wrap_w;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, max_h - wrap_h, center_x - half_w, 0, 0, wrap_h, width_before_wrap, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, max_h - wrap_h, 0, 0, width_before_wrap, wrap_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, center_x - half_w, wrap_h, 0, center_y + half_h, width_before_wrap, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, 0, wrap_h, width_before_wrap, center_y + half_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 4: 左下角
        else if (center_x < yaw_min && center_y > pitch_max) {
            int wrap_w = yaw_min - center_x; int wrap_h = center_y - pitch_max; int height_before_wrap = chunk_h - wrap_h;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, max_w - wrap_w, 0, 0, height_before_wrap, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, 0, 0, wrap_w, height_before_wrap, center_x + half_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, max_w - wrap_w, height_before_wrap, 0, wrap_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, 0, height_before_wrap, wrap_w, wrap_h, center_x + half_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 5: 右下角
        else if (center_x > yaw_max && center_y > pitch_max) {
            int wrap_w = center_x - yaw_max; int wrap_h = center_y - pitch_max; int width_before_wrap = chunk_w - wrap_w; int height_before_wrap = chunk_h - wrap_h;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, center_x - half_w, 0, 0, height_before_wrap, width_before_wrap, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, 0, 0, width_before_wrap, height_before_wrap, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, center_x - half_w, height_before_wrap, 0, wrap_h, width_before_wrap, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, 0, height_before_wrap, width_before_wrap, wrap_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 6: 左边界
        else if (center_x < yaw_min) {
            int wrap_w = yaw_min - center_x;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, max_w - wrap_w, 0, 0, chunk_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, 0, 0, wrap_w, chunk_h, center_x + half_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 7: 右边界
        else if (center_x > yaw_max) {
            int wrap_w = center_x - yaw_max; int width_before_wrap = chunk_w - wrap_w;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, center_x - half_w, 0, 0, chunk_h, width_before_wrap, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, 0, 0, width_before_wrap, chunk_h, wrap_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 8: 上边界
        else if (center_y < pitch_min) {
            int wrap_h = pitch_min - center_y;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, max_h - wrap_h, center_x - half_w, 0, 0, wrap_h, chunk_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, center_x - half_w, wrap_h, 0, center_y + half_h, chunk_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
        // Case 9: 下边界
        else { // center_y > pitch_max
            int wrap_h = center_y - pitch_max; int height_before_wrap = chunk_h - wrap_h;
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, center_y - half_h, center_x - half_w, 0, 0, height_before_wrap, chunk_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
            safe_copy_with_border(out_ptr, chunk_ptr, actual_chunk_length, 3, 0, center_x - half_w, height_before_wrap, 0, wrap_h, chunk_w, max_h, max_w, chunk_h, chunk_w, chunk_strides, border_thickness);
        }
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error in paste_viewport_to_video: ") + e.what());
    }

    return output;
}

// ============================================================================
//
//                      pybind11 模块定义
//
// ============================================================================
PYBIND11_MODULE(viewport_ops_cpp, m) {
    m.doc() = "C++ accelerated viewport operations with OpenMP and stride-aware safety checks";

    m.def("real_gen_viewport_chunks", &real_gen_viewport_chunks_cpp,
          "Generates the video chunk for transmission (C++, Stride-Aware).",
          py::arg("video_frames"), py::arg("total_frames"), py::arg("angel"),
          py::arg("indx"), py::arg("max_size"), py::arg("chunk_length"),
          py::arg("chunk_size")
    );

    m.def("paste_viewport_to_video", &paste_viewport_to_video_cpp,
          "Pastes the viewport chunk back to the video (C++, Stride-Aware).",
          py::arg("video_frames"), py::arg("viewport_chunk"), py::arg("angel"),
          py::arg("indx"), py::arg("max_size"), py::arg("chunk_length"),
          py::arg("chunk_size"), py::arg("border_thickness")
    );

#ifdef _OPENMP
    m.def("get_num_threads", []() { return omp_get_max_threads(); }, "Get the number of OpenMP threads available");
    m.def("set_num_threads", [](int n_threads) {
        SAFE_CHECK(n_threads > 0, "Number of threads must be positive");
        omp_set_num_threads(n_threads);
    }, "Set the number of OpenMP threads to use", py::arg("n_threads"));
#else
    m.def("get_num_threads", []() { return 1; });
    m.def("set_num_threads", [](int) { /* do nothing */ });
#endif

    m.attr("__version__") = "1.2.1-compilefix";
}
