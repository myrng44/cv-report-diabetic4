"""
Module tiền xử lý: CLAHE Enhancement cho phát hiện bệnh lý võng mạc tiểu đường
"""

import cv2
import numpy as np
from typing import Tuple


class CLAHEEnhancer:
    """
    Contrast Limited Adaptive Histogram Equalization được tối ưu hóa
    Triển khai tùy chỉnh với các tối ưu hóa hiệu suất
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Khởi tạo bộ tăng cường CLAHE
        
        Args:
            clip_limit: Ngưỡng giới hạn độ tương phản (khuyến nghị 2.0-4.0)
            tile_grid_size: Kích thước lưới cho cân bằng histogram (mặc định 8x8)
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        # Tạo trước đối tượng CLAHE để tăng hiệu suất
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng CLAHE để tăng cường độ tương phản
        
        Đối với ảnh màu: Chuyển sang LAB và chỉ áp dụng cho kênh L
        Đối với ảnh grayscale: Áp dụng trực tiếp
        
        Args:
            image: Ảnh đầu vào (BGR hoặc grayscale)
            
        Returns:
            Ảnh đã tăng cường với cùng shape và dtype như đầu vào
        """
        if len(image.shape) == 3:
            # Chuyển sang không gian màu LAB để bảo toàn màu sắc tốt hơn
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Chỉ áp dụng CLAHE cho kênh L (độ sáng)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            
            # Chuyển lại sang BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            # Áp dụng trực tiếp cho ảnh grayscale
            return self.clahe.apply(image)
    
    def apply_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng CLAHE cho ảnh RGB (giả định đầu vào là RGB, không phải BGR)
        
        Args:
            image: Ảnh RGB đầu vào
            
        Returns:
            Ảnh RGB đã tăng cường
        """
        if len(image.shape) == 3:
            # Chuyển RGB sang LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced
        else:
            return self.clahe.apply(image)


class ImagePreprocessor:
    """Pipeline tiền xử lý ảnh đơn giản với CLAHE"""

    def __init__(self, target_size: int = 256, clip_limit: float = 2.0):
        self.target_size = target_size
        self.clahe_enhancer = CLAHEEnhancer(clip_limit=clip_limit)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline tiền xử lý:
        1. Thay đổi kích thước
        2. Tăng cường độ tương phản bằng CLAHE
        """
        # Thay đổi kích thước ảnh
        image = cv2.resize(image, (self.target_size, self.target_size))

        # Áp dụng CLAHE để tăng cường độ tương phản
        enhanced = self.clahe_enhancer.apply(image)

        return enhanced

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Chuẩn hóa ảnh về khoảng [0, 1]"""
        return image.astype(np.float32) / 255.0


def preprocess_fundus_image(image_path: str, target_size: int = 256,
                            clip_limit: float = 2.0) -> np.ndarray:
    """
    Hàm tiện ích để tiền xử lý ảnh đáy mắt từ file
    Áp dụng tăng cường CLAHE để cải thiện độ tương phản
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Tiền xử lý
    preprocessor = ImagePreprocessor(target_size=target_size, clip_limit=clip_limit)
    processed = preprocessor.preprocess(image)
    normalized = preprocessor.normalize(processed)

    return normalized
