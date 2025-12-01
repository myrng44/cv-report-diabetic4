"""
Module tiền xử lý: CLAHE Enhancement cho phát hiện bệnh lý võng mạc tiểu đường
Triển khai CLAHE thủ công
"""

import cv2
import numpy as np
from typing import Tuple


class CLAHEEnhancer:
    """
    Contrast Limited Adaptive Histogram Equalization - Triển khai thủ công
    
    Thuật toán CLAHE:
    1. Chia ảnh thành các tile (vùng nhỏ)
    2. Tính histogram cho mỗi tile
    3. Áp dụng clip limit để giới hạn độ tương phản
    4. Cân bằng histogram (equalization)
    5. Nội suy bilinear giữa các tile để tránh artifact
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
    
    def _compute_histogram(self, tile: np.ndarray) -> np.ndarray:
        """
        Tính histogram cho 1 tile
        
        Args:
            tile: Tile ảnh grayscale
            
        Returns:
            Histogram 256 bins
        """
        hist = np.zeros(256, dtype=np.float32)
        for pixel_value in tile.flatten():
            hist[pixel_value] += 1
        return hist
    
    def _clip_histogram(self, hist: np.ndarray, clip_limit: float) -> np.ndarray:
        """
        Áp dụng clip limit để giới hạn độ tương phản
        
        Args:
            hist: Histogram gốc
            clip_limit: Ngưỡng clip
            
        Returns:
            Histogram đã clip
        """
        # Tính số pixels bị cắt
        hist_clipped = np.copy(hist)
        
        # Tính ngưỡng clip tuyệt đối
        total_pixels = np.sum(hist)
        clip_value = (total_pixels / 256) * self.clip_limit
        
        # Clip histogram
        excess = 0
        for i in range(256):
            if hist_clipped[i] > clip_value:
                excess += hist_clipped[i] - clip_value
                hist_clipped[i] = clip_value
        
        # Phân phối lại excess pixels đều cho tất cả bins
        redistribute = excess / 256
        hist_clipped += redistribute
        
        return hist_clipped
    
    def _equalize_histogram(self, hist: np.ndarray) -> np.ndarray:
        """
        Cân bằng histogram (histogram equalization)
        
        Args:
            hist: Histogram đã clip
            
        Returns:
            Lookup table (LUT) để map giá trị cũ -> mới
        """
        # Tính CDF (Cumulative Distribution Function)
        cdf = np.cumsum(hist)
        
        # Normalize CDF về [0, 255]
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
        cdf_max = cdf.max()
        
        if cdf_max - cdf_min == 0:
            return np.arange(256, dtype=np.uint8)
        
        # Tạo lookup table
        lut = ((cdf - cdf_min) / (cdf_max - cdf_min) * 255).astype(np.uint8)
        
        return lut
    
    def _bilinear_interpolate(self, tile_luts: np.ndarray, img: np.ndarray, 
                             tile_h: int, tile_w: int) -> np.ndarray:
        """
        Nội suy bilinear giữa các tile để tránh block artifacts
        
        Args:
            tile_luts: Lookup tables của tất cả tiles
            img: Ảnh gốc
            tile_h: Chiều cao mỗi tile
            tile_w: Chiều rộng mỗi tile
            
        Returns:
            Ảnh đã được nội suy
        """
        h, w = img.shape
        result = np.zeros_like(img)
        
        n_tiles_y, n_tiles_x = self.tile_grid_size
        
        for y in range(h):
            for x in range(w):
                # Tìm vị trí tile
                tile_y = min(y / tile_h, n_tiles_y - 1)
                tile_x = min(x / tile_w, n_tiles_x - 1)
                
                # Tile indices
                ty0 = int(np.floor(tile_y))
                ty1 = min(ty0 + 1, n_tiles_y - 1)
                tx0 = int(np.floor(tile_x))
                tx1 = min(tx0 + 1, n_tiles_x - 1)
                
                # Trọng số nội suy
                wy = tile_y - ty0
                wx = tile_x - tx0
                
                # Lấy giá trị pixel gốc
                pixel_value = img[y, x]
                
                # Nội suy từ 4 tile xung quanh
                v00 = tile_luts[ty0, tx0][pixel_value]
                v01 = tile_luts[ty0, tx1][pixel_value]
                v10 = tile_luts[ty1, tx0][pixel_value]
                v11 = tile_luts[ty1, tx1][pixel_value]
                
                # Bilinear interpolation
                v0 = v00 * (1 - wx) + v01 * wx
                v1 = v10 * (1 - wx) + v11 * wx
                result[y, x] = v0 * (1 - wy) + v1 * wy
        
        return result.astype(np.uint8)
    
    def _apply_clahe_manual(self, img: np.ndarray) -> np.ndarray:
        """
        Áp dụng CLAHE thủ công lên ảnh grayscale
        
        Args:
            img: Ảnh grayscale
            
        Returns:
            Ảnh đã tăng cường
        """
        h, w = img.shape
        n_tiles_y, n_tiles_x = self.tile_grid_size
        
        # Tính kích thước mỗi tile
        tile_h = h // n_tiles_y
        tile_w = w // n_tiles_x
        
        # Lưu LUT cho mỗi tile
        tile_luts = np.zeros((n_tiles_y, n_tiles_x, 256), dtype=np.uint8)
        
        # Xử lý từng tile
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Cắt tile
                y1 = ty * tile_h
                y2 = (ty + 1) * tile_h if ty < n_tiles_y - 1 else h
                x1 = tx * tile_w
                x2 = (tx + 1) * tile_w if tx < n_tiles_x - 1 else w
                
                tile = img[y1:y2, x1:x2]
                
                # Tính histogram
                hist = self._compute_histogram(tile)
                
                # Clip histogram
                hist_clipped = self._clip_histogram(hist, self.clip_limit)
                
                # Equalize histogram
                lut = self._equalize_histogram(hist_clipped)
                
                # Lưu LUT
                tile_luts[ty, tx] = lut
        
        # Nội suy bilinear giữa các tile
        result = self._bilinear_interpolate(tile_luts, img, tile_h, tile_w)
        
        return result

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
            
            # Chỉ áp dụng CLAHE cho kênh L (độ sáng) - TRIỂN KHAI THỦ CÔNG
            lab[:, :, 0] = self._apply_clahe_manual(lab[:, :, 0])
            
            # Chuyển lại sang BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            # Áp dụng trực tiếp cho ảnh grayscale - TRIỂN KHAI THỦ CÔNG
            return self._apply_clahe_manual(image)
    
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
            # Áp dụng CLAHE thủ công cho kênh L
            lab[:, :, 0] = self._apply_clahe_manual(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced
        else:
            return self._apply_clahe_manual(image)


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
