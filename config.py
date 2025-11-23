"""
Configuration file for SANGO-based Diabetic Retinopathy Detection System
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Đường dẫn phân loại
CLASS_TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'B. Disease Grading', '1. Original Images', 'a. Training Set')
CLASS_TEST_IMG_DIR = os.path.join(DATA_DIR, 'B. Disease Grading', '1. Original Images', 'b. Testing Set')
CLASS_TRAIN_LABELS = os.path.join(DATA_DIR, 'B. Disease Grading', '2. Groundtruths', 'a. IDRiD_Disease Grading_Training Labels.csv')
CLASS_TEST_LABELS = os.path.join(DATA_DIR, 'B. Disease Grading', '2. Groundtruths', 'b. IDRiD_Disease Grading_Testing Labels.csv')

# Đường dẫn phân đoạn
SEG_TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '1. Original Images', 'a. Training Set')
SEG_TEST_IMG_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '1. Original Images', 'b. Testing Set')
SEG_TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '2. All Segmentation Groundtruths', 'a. Training Set')
SEG_TEST_MASK_DIR = os.path.join(DATA_DIR, 'A. Segmentation', '2. All Segmentation Groundtruths', 'b. Testing Set')

# Thư mục đầu ra
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')

# Tạo các thư mục đầu ra
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Siêu tham số mô hình - Được tối ưu hóa để có hiệu suất tốt hơn
IMG_SIZE = 768  # Cho phân loại - độ phân giải cao hơn
SEG_IMG_SIZE = 1024  # Cho phân đoạn - PHẢI cao cho các tổn thương nhỏ
BATCH_SIZE = 8  # Cân bằng cho T4 GPU
NUM_WORKERS = 2  # Giảm để ổn định
NUM_CLASSES = 5  # Mức độ DR: 0, 1, 2, 3, 4
SEG_CLASSES = 3  # Microaneurysms, Haemorrhages, Hard Exudates

# Tham số huấn luyện - Được tối ưu hóa
LEARNING_RATE = 1e-4  # Phân loại
SEG_LEARNING_RATE = 5e-5  # Phân đoạn - thấp hơn để ổn định
NUM_EPOCHS = 150  # Nhiều epoch hơn để hội tụ
EARLY_STOPPING_PATIENCE = 35  # Kiên nhẫn lâu hơn
WEIGHT_DECAY = 3e-4  # Regularization mạnh hơn

# Tham số thuật toán SANGO
POPULATION_SIZE = 20  # Giảm để tính toán nhanh hơn
MAX_ITERATIONS = 30  # Giảm để huấn luyện nhanh hơn
ALPHA = 2.0  # Tham số tự thích nghi
BETA = 1.5  # Tham số khám phá Northern Goshawk

# Tham số GRU
GRU_HIDDEN_SIZE = 128  # Giảm từ 256
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.3

# Tham số CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Tăng cường dữ liệu (chỉ cho huấn luyện)
AUGMENTATION = True
ROTATION_RANGE = 15
BRIGHTNESS_RANGE = 0.2
CONTRAST_RANGE = 0.2

# Cấu hình thiết bị
DEVICE = 'cuda'  # Sẽ tự động phát hiện trong code

# Huấn luyện độ chính xác hỗn hợp để tiết kiệm bộ nhớ
USE_AMP = True  # Automatic Mixed Precision

# Tích lũy gradient cho batch size hiệu quả lớn hơn
GRADIENT_ACCUMULATION_STEPS = 2

# Seed ngẫu nhiên để tái tạo kết quả
SEED = 42

# Chiến lược huấn luyện nâng cao (Giai đoạn 1 - Chiến thắng nhanh!)
USE_CLASS_WEIGHTS = False  # Giữ tắt để ổn định
USE_MIXUP = True  # Bật với trộn nhẹ nhàng
MIXUP_ALPHA = 0.1  # Giảm từ 0.2 để tăng cường nhẹ nhàng hơn
USE_LABEL_SMOOTHING = True  # Bật để giảm overfitting
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0
USE_COSINE_SCHEDULE = False  # Giữ ReduceLROnPlateau để ổn định
WARMUP_EPOCHS = 5
USE_TTA = False  # Tắt cho huấn luyện, bật cho suy luận

# Tham số OneCycleLR (cho phân đoạn)
MAX_LR = 1e-4  # Learning rate tối đa cho OneCycleLR
PCT_START = 0.3  # Phần trăm huấn luyện cho warm-up

# Kỹ thuật huấn luyện nâng cao
USE_MIXUP = True
MIXUP_ALPHA = 0.2  # Tăng để tăng cường nhiều hơn
USE_CUTMIX = True
CUTMIX_ALPHA = 1.0
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.1

# Đặc biệt cho phân đoạn
USE_DEEP_SUPERVISION = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DICE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.3
TVERSKY_WEIGHT = 0.2
