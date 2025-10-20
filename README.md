# SANGO-based Diabetic Retinopathy Detection System

## Giới thiệu

Hệ thống phát hiện và phân loại bệnh võng mạc tiểu đường (Diabetic Retinopathy) sử dụng thuật toán **SANGO (Self-Adaptive Northern Goshawk Optimization)** dựa trên bài báo khoa học.

### Tính năng chính:

1. **Classification**: Phân loại mức độ nghiêm trọng của DR (5 cấp độ: 0-4)
   - 0: No DR
   - 1: Mild NPDR
   - 2: Moderate NPDR
   - 3: Severe NPDR
   - 4: PDR

2. **Segmentation**: Phân đoạn các vùng tổn thương võng mạc
   - Microaneurysms (MA)
   - Haemorrhages (HEM)
   - Hard Exudates (EX)

3. **Explainable AI**: Sử dụng Grad-CAM để visualize vùng mà model tập trung

## Kiến trúc hệ thống

### 1. Preprocessing
- **Adaptive Gabor Filter** với Chebyshev Chaotic Map: Giảm nhiễu
- **CLAHE**: Tăng cường độ tương phản

### 2. Segmentation Model
- **Modified U-Net** với Adaptive Batch Normalization
- EfficientNet layers cho encoder
- Combined Loss (BCE + Dice Loss)

### 3. Classification Model
- **DenseNet121** backbone (pretrained)
- **Multi-folded Feature Extraction**: LBP + SURF + TEM
- **Attention Mechanism**
- **Optimized GRU** với SANGO optimization
- **Focal Loss** để xử lý class imbalance

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA 11.0+ (khuyến nghị cho GPU)
- RAM: 8GB+ (16GB khuyến nghị)
- GPU: 4GB+ VRAM (khuyến nghị)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Cấu trúc dữ liệu

```
data/
├── A. Segmentation/
│   ├── 1. Original Images/
│   │   ├── a. Training Set/
│   │   └── b. Testing Set/
│   └── 2. All Segmentation Groundtruths/
│       └── a. Training Set/
│           ├── 1. Microaneurysms/
│           ├── 2. Haemorrhages/
│           └── 3. Hard Exudates/
└── B. Disease Grading/
    ├── 1. Original Images/
    │   ├── a. Training Set/
    │   └── b. Testing Set/
    └── 2. Groundtruths/
        ├── a. IDRiD_Disease Grading_Training Labels.csv
        └── b. IDRiD_Disease Grading_Testing Labels.csv
```

## Sử dụng

### 1. Training Classification Model

```bash
python main.py --mode train_classification --epochs 50 --batch_size 4
```

### 2. Training Segmentation Model

```bash
python main.py --mode train_segmentation --epochs 50 --batch_size 4
```

### 3. Training cả 2 models

```bash
python main.py --mode train_all --epochs 50 --batch_size 4
```

### 4. Inference trên ảnh mới

```bash
python main.py --mode inference --image_path path/to/image.jpg
```

## Tối ưu hóa bộ nhớ

Hệ thống đã được tối ưu hóa để chạy trên GPU với bộ nhớ hạn chế:

1. **Image Size**: 256x256 (có thể giảm xuống 224 nếu cần)
2. **Batch Size**: 4 (tối ưu cho 4-6GB VRAM)
3. **Mixed Precision Training**: Sử dụng AMP để giảm 50% memory
4. **Gradient Accumulation**: Hiệu quả batch size lớn hơn
5. **DenseNet121**: Model nhẹ hơn so với DenseNet201
6. **Base Filters**: 32 cho U-Net (có thể giảm xuống 16)

### Nếu gặp lỗi Out of Memory (OOM):

1. Giảm batch size trong `config.py`:
```python
BATCH_SIZE = 2  # hoặc 1
```

2. Giảm image size:
```python
IMG_SIZE = 224  # hoặc 192
```

3. Giảm base filters của U-Net:
```python
# Trong train_segmentation.py
model = create_segmentation_model(base_filters=16)
```

4. Tắt data augmentation nặng:
```python
AUGMENTATION = False
```

## Cấu hình hyperparameters

Chỉnh sửa file `config.py` để thay đổi các tham số:

```python
# Image and training params
IMG_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# SANGO params
POPULATION_SIZE = 20
MAX_ITERATIONS = 30

# GRU params
GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS = 2
```

## Kết quả

Models và kết quả sẽ được lưu trong thư mục `outputs/`:

```
outputs/
├── models/          # Trained models
│   ├── best_model.pth              # Best classification model
│   ├── best_seg_model.pth          # Best segmentation model
│   └── checkpoint_epoch_*.pth      # Checkpoints
├── logs/            # Training logs
└── results/         # Visualizations
    ├── classification_metrics.png
    ├── segmentation_metrics.png
    └── inference_*.jpg
```

## Đánh giá hiệu suất

### Classification Metrics:
- Accuracy
- F1 Score
- Precision
- Recall
- Confusion Matrix

### Segmentation Metrics:
- IoU (Intersection over Union)
- Dice Similarity Coefficient
- Pixel Accuracy

## Thuật toán SANGO

**Self-Adaptive Northern Goshawk Optimization** được sử dụng để tối ưu hóa hyperparameters của GRU:

- Tự động tìm hidden size tối ưu
- Tự động tìm dropout rate tốt nhất
- Balance giữa exploration và exploitation
- Sử dụng Levy flight cho global search

## Tips để tăng hiệu suất

1. **Data Augmentation**: Bật augmentation trong training để tránh overfitting
2. **Early Stopping**: Patience = 10 epochs để tránh overtraining
3. **Learning Rate Scheduling**: Sử dụng ReduceLROnPlateau hoặc CosineAnnealing
4. **Mixed Precision**: Luôn bật AMP để tăng tốc độ và giảm memory
5. **Pretrained Weights**: Sử dụng ImageNet pretrained cho DenseNet

## Troubleshooting

### GPU không được sử dụng
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Lỗi CUDA Out of Memory
- Giảm batch size
- Giảm image size
- Tắt một số data augmentation

### Training quá chậm
- Tăng num_workers trong DataLoader (nhưng cẩn thận với RAM)
- Sử dụng smaller model
- Giảm số epochs

## Citation

Nếu sử dụng code này, vui lòng cite bài báo gốc:

```
Sharma, N., & Lalwani, P. (2025). 
A multi model deep net with an explainable AI based framework 
for diabetic retinopathy segmentation and classification. 
Scientific Reports, 15, 8777.
```

## License

MIT License - Xem file LICENSE để biết thêm chi tiết.

## Contact

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue trên GitHub.

