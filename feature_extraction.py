"""
Multi-folded feature extraction: LBP, SURF, and TEM (Texture Energy Measurement)
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from typing import Tuple


class LBPFeatureExtractor:
    """Local Binary Pattern feature extraction"""

    def __init__(self, n_points: int = 8, radius: int = 1):
        self.n_points = n_points
        self.radius = radius

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract LBP features"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute LBP
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')

        # Compute histogram
        n_bins = self.n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        return hist


class SURFFeatureExtractor:
    """SURF (Speeded-Up Robust Features) extraction"""

    def __init__(self, hessian_threshold: int = 400, n_features: int = 100):
        self.hessian_threshold = hessian_threshold
        self.n_features = n_features
        # Note: SURF is patented, using ORB as alternative for production
        self.detector = cv2.ORB_create(nfeatures=n_features)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract SURF-like features using ORB"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        if descriptors is None or len(descriptors) == 0:
            # Return zero vector if no features detected
            return np.zeros(self.n_features * 32)

        # Flatten and pad/truncate to fixed size
        flat_desc = descriptors.flatten()
        target_size = self.n_features * 32

        if len(flat_desc) < target_size:
            # Pad with zeros
            flat_desc = np.pad(flat_desc, (0, target_size - len(flat_desc)))
        else:
            # Truncate
            flat_desc = flat_desc[:target_size]

        return flat_desc


class TEMFeatureExtractor:
    """Texture Energy Measurement feature extraction"""

    def __init__(self, n_scales: int = 3, n_orientations: int = 4):
        self.n_scales = n_scales
        self.n_orientations = n_orientations

    def _gabor_kernel(self, ksize: int, sigma: float, theta: float, lambd: float, gamma: float) -> np.ndarray:
        """Create Gabor kernel"""
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=0,
            ktype=cv2.CV_32F
        )
        return kernel

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract TEM features using Gabor filters"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = []

        # Multi-scale, multi-orientation Gabor filtering
        for scale in range(self.n_scales):
            sigma = 2 ** scale
            lambd = sigma * 1.5

            for orient in range(self.n_orientations):
                theta = np.pi * orient / self.n_orientations

                # Create and apply Gabor kernel
                kernel = self._gabor_kernel(21, sigma, theta, lambd, 0.5)
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)

                # Compute energy (mean and std of response)
                energy = [np.mean(filtered), np.std(filtered)]
                features.extend(energy)

        return np.array(features)


class MultiFeatureExtractor:
    """Combined multi-folded feature extraction"""

    def __init__(self):
        self.lbp_extractor = LBPFeatureExtractor()
        self.surf_extractor = SURFFeatureExtractor(n_features=50)  # Reduced for memory
        self.tem_extractor = TEMFeatureExtractor()

    def extract_all(self, image: np.ndarray) -> np.ndarray:
        """Extract all features and concatenate"""
        lbp_features = self.lbp_extractor.extract(image)
        surf_features = self.surf_extractor.extract(image)
        tem_features = self.tem_extractor.extract(image)

        # Concatenate all features
        all_features = np.concatenate([lbp_features, surf_features, tem_features])

        return all_features

    def get_feature_dim(self) -> int:
        """Get total feature dimension"""
        # LBP: 10 (8 points + 2)
        # SURF: 50 * 32 = 1600
        # TEM: 3 scales * 4 orientations * 2 = 24
        return 10 + 1600 + 24  # = 1634


class DenseNetFeatureExtractor(cv2.dnn.Net):
    """DenseNet-based feature extraction using pretrained model"""

    def __init__(self):
        super().__init__()
        # This is a placeholder - in practice, use torchvision.models.densenet
        self.model = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features using DenseNet"""
        # This will be implemented in the main classification model
        pass


def extract_features_from_image(image: np.ndarray) -> np.ndarray:
    """
    Convenience function to extract all features from an image

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        Feature vector
    """
    extractor = MultiFeatureExtractor()
    features = extractor.extract_all(image)
    return features

