"""
Preprocessing module: Adaptive Gabor Filter with Chaotic Map + CLAHE
"""

import cv2
import numpy as np
from typing import Tuple

class ChebyshevChaoticMap:
    """Chebyshev chaotic map for adaptive parameter tuning"""

    def __init__(self, initial_value: float = 0.7, order: int = 5):
        self.value = initial_value
        self.order = order

    def next(self) -> float:
        """Generate next chaotic value using Chebyshev map"""
        self.value = np.cos(self.order * np.arccos(self.value))
        return self.value


class AdaptiveGaborFilter:
    """Adaptive Gabor Filter with Chaotic Map enhancement"""

    def __init__(self, ksize: int = 31, sigma: float = 3.0, lambd: float = 10.0,
                 gamma: float = 0.5, n_orientations: int = 8):
        self.ksize = ksize
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        self.n_orientations = n_orientations
        self.chaotic_map = ChebyshevChaoticMap()

    def get_gabor_kernel(self, theta: float, chaotic_value: float) -> np.ndarray:
        """Generate adaptive Gabor kernel with chaotic enhancement"""
        # Add chaotic perturbation to theta
        adaptive_theta = theta + chaotic_value * 0.1

        # Create Gabor kernel
        kernel = cv2.getGaborKernel(
            ksize=(self.ksize, self.ksize),
            sigma=self.sigma,
            theta=adaptive_theta,
            lambd=self.lambd,
            gamma=self.gamma,
            psi=0,
            ktype=cv2.CV_32F
        )

        # Normalize
        kernel = kernel / kernel.sum()
        return kernel

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive Gabor filtering to image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize output
        filtered = np.zeros_like(image, dtype=np.float32)

        # Apply Gabor filters with different orientations
        for i in range(self.n_orientations):
            theta = np.pi * i / self.n_orientations
            chaotic_value = self.chaotic_map.next()

            # Get adaptive kernel
            kernel = self.get_gabor_kernel(theta, chaotic_value)

            # Apply filter
            response = cv2.filter2D(image, cv2.CV_32F, kernel)
            filtered += np.abs(response)

        # Normalize to [0, 255]
        filtered = filtered / self.n_orientations
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        return filtered


class CLAHEEnhancer:
    """Contrast Limited Adaptive Histogram Equalization"""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance contrast"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Apply CLAHE to L channel
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            return self.clahe.apply(image)


class ImagePreprocessor:
    """Complete image preprocessing pipeline"""

    def __init__(self, target_size: int = 256):
        self.target_size = target_size
        self.gabor_filter = AdaptiveGaborFilter()
        self.clahe_enhancer = CLAHEEnhancer()

    def preprocess(self, image: np.ndarray, apply_gabor: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline:
        1. Resize
        2. Adaptive Gabor filtering (optional, for denoising)
        3. CLAHE contrast enhancement
        """
        # Resize image
        image = cv2.resize(image, (self.target_size, self.target_size))

        # Apply Gabor filter for denoising (optional)
        if apply_gabor:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            filtered = self.gabor_filter.apply(gray)
            # Convert back to BGR
            if len(image.shape) == 3:
                image = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            else:
                image = filtered

        # Apply CLAHE for contrast enhancement
        enhanced = self.clahe_enhancer.apply(image)

        return enhanced

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0


def preprocess_fundus_image(image_path: str, target_size: int = 256,
                            apply_gabor: bool = False) -> np.ndarray:
    """
    Convenience function to preprocess a fundus image from file
    Note: Gabor filter is computationally expensive, set to False by default
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Preprocess
    preprocessor = ImagePreprocessor(target_size=target_size)
    processed = preprocessor.preprocess(image, apply_gabor=apply_gabor)
    normalized = preprocessor.normalize(processed)

    return normalized

