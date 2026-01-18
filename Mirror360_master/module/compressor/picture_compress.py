import os
import time
import uuid
from PIL import Image
import tempfile

class SimpleJPEGCompressor:
    """
    仅使用标准 JPEG 有损压缩的“压缩器”和“解压器”
    （本质是保存为JPEG → 读取JPEG，JPEG本身就是有损格式）
    """
    @staticmethod
    def compress(image: Image.Image, 
                 quality: int = 60, 
                 optimize: bool = True,
                 temp_dir: str = None) -> dict:

        start_time = time.time()
        
        # 估算原始图像内存大小（近似值）
        if image.mode in ('RGBA', 'LA'):
            bytes_per_pixel = 4
        elif image.mode == 'RGB':
            bytes_per_pixel = 3
        elif image.mode in ('L', '1'):
            bytes_per_pixel = 1
        else:
            image = image.convert('RGB')
            bytes_per_pixel = 3
        original_size_estimate = image.width * image.height * bytes_per_pixel
        
        # 创建临时文件
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
            
        compressed_path = os.path.join(temp_dir, f"compressed_{uuid.uuid4().hex}.jpg")
        
        # 处理透明通道 → 白色背景（与你原函数一致）
        save_img = image.copy()
        save_img = save_img.convert('RGB')  # 强制转RGB
        
        # 保存为JPEG（这就是“压缩”过程）
        save_img.save(compressed_path, 'JPEG', quality=quality, optimize=optimize)
        
        compressed_size = os.path.getsize(compressed_path)
        compression_ratio = (1 - compressed_size / max(original_size_estimate, 1)) * 100
        elapsed = time.time() - start_time
        
        return compressed_path,compressed_size 
    @staticmethod
    def decompress(compressed_path: str) -> Image.Image:
        """
        解压：读取压缩后的 JPEG 文件，尝试还原图像（有损，无法100%还原）
        
        Args:
            compressed_path: compress() 返回的 JPEG 文件路径
            
        Returns:
            PIL.Image.Image 对象（RGB模式，接近原始图像）
        """
        if not os.path.exists(compressed_path):
            raise FileNotFoundError(f"压缩文件不存在: {compressed_path}")
            
        # 直接打开JPEG就是“解压”过程
        restored_image = Image.open(compressed_path)
        restored_image = restored_image.convert('RGB')  # 确保输出是RGB
        
        return restored_image