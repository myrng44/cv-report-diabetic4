"""
Configuration file for SANGO-based Diabetic Retinopathy Detection System
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Classification paths
CLASS_TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'B. Disease Grading', '1. Original Images', 'a. Training Set')
CLASS_TEST_IMG_DIR = os.path.join(DATA_DIR, 'B. Disease Grading', '1. Original Images', 'b. Testing Set')
CLASS_TRAIN_LABELS = os.path.join(DATA_DIR, 'B. Disease Grading', '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
CLASS_TEST_LABELS = os.path.join(DATA_DIR, 'B. Disease Grading', '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')

# Segmentation paths
SEG_TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '1. Original Images', 'a. Training Set')
SEG_TEST_IMG_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '1. Original Images', 'b. Testing Set')
SEG_TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '2. All Segmentation Groundtruths', 'a. Training Set')
SEG_TEST_MASK_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '2. All Segmentation Groundtruths', 'b. Testing Set')

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')

# Create output directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model hyperparameters - Optimized for better performance
IMG_SIZE = 384  # Increased from 256 for better feature extraction
BATCH_SIZE = 4  # Keep at 4 for CPU, can increase to 8-16 with GPU
NUM_WORKERS = 2  # Reduced for stability
NUM_CLASSES = 5  # DR grades: 0, 1, 2, 3, 4
SEG_CLASSES = 3  # Microaneurysms, Haemorrhages, Hard Exudates

# Training parameters
LEARNING_RATE = 5e-5  # Reduced for more stable training
NUM_EPOCHS = 50  # Increased for better convergence
EARLY_STOPPING_PATIENCE = 15  # Increased patience
WEIGHT_DECAY = 1e-4  # Increased for better regularization

# SANGO algorithm parameters
POPULATION_SIZE = 20  # Reduced for faster computation
MAX_ITERATIONS = 30  # Reduced for faster training
ALPHA = 2.0  # Self-adaptive parameter
BETA = 1.5  # Northern Goshawk exploration parameter

# GRU parameters
GRU_HIDDEN_SIZE = 128  # Reduced from 256
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.3

# Feature extraction parameters
LBP_POINTS = 8
LBP_RADIUS = 1
SURF_THRESHOLD = 100

# Gabor filter parameters
GABOR_FREQUENCY = 0.1
GABOR_THETA_VALUES = 8  # Number of orientations
GABOR_SIGMA = 3
GABOR_LAMBDA = 10
GABOR_GAMMA = 0.5

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Data augmentation (for training only)
AUGMENTATION = True
ROTATION_RANGE = 15
BRIGHTNESS_RANGE = 0.2
CONTRAST_RANGE = 0.2

# Device configuration
DEVICE = 'cuda'  # Will auto-detect in code

# Mixed precision training for memory efficiency
USE_AMP = True  # Automatic Mixed Precision

# Gradient accumulation for effective larger batch size
GRADIENT_ACCUMULATION_STEPS = 2

# Random seed for reproducibility
SEED = 42

# Advanced training strategies (from advanced_strategies.py)
USE_CLASS_WEIGHTS = True  # Important for imbalanced dataset!
USE_MIXUP = True  # Very effective for small datasets
MIXUP_ALPHA = 0.2
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0
USE_COSINE_SCHEDULE = True  # Better than ReduceLROnPlateau
WARMUP_EPOCHS = 5
USE_TTA = False  # Disable for training, enable for inference
