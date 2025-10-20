# Quick Start Guide - SANGO DR Detection System

## 🚀 Hướng dẫn nhanh

### Bước 1: Kiểm tra hệ thống

```bash
python test_system.py
```

✅ Đảm bảo ít nhất 5/7 tests pass (CUDA test có thể fail nếu không có GPU)

### Bước 2: Phân tích dữ liệu (Tùy chọn)

```bash
python utils.py
```

Script này sẽ:
- Phân tích phân bố classes
- Kiểm tra kích thước ảnh
- Ước tính thời gian training
- Tạo visualization của dữ liệu

### Bước 3: Training

#### Option A: Train Classification Only (Nhanh hơn)
```bash
python main.py --mode train_classification --epochs 30 --batch_size 4
```

#### Option B: Train Segmentation Only
```bash
python main.py --mode train_segmentation --epochs 30 --batch_size 4
```

#### Option C: Train Both (Recommended)
```bash
python main.py --mode train_all --epochs 30 --batch_size 4
```

### Bước 4: Đánh giá model

```bash
python evaluate.py
```

### Bước 5: Inference trên ảnh mới

```bash
python main.py --mode inference --image_path "data/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg"
```

## 📊 Tham số có thể điều chỉnh

### Nếu gặp Out of Memory:

**Trong file `config.py`:**
```python
IMG_SIZE = 224          # Giảm từ 256
BATCH_SIZE = 2          # Giảm từ 4
GRU_HIDDEN_SIZE = 64    # Giảm từ 128
```

**Trong `train_segmentation.py`:**
```python
model = create_segmentation_model(base_filters=16)  # Giảm từ 32
```

### Để training nhanh hơn (nhưng accuracy thấp hơn):

```python
NUM_EPOCHS = 20         # Giảm từ 50
POPULATION_SIZE = 10    # Giảm từ 20 (SANGO)
MAX_ITERATIONS = 15     # Giảm từ 30 (SANGO)
```

## 🎯 Expected Results

### Classification:
- **Accuracy**: 85-95% (tùy theo dữ liệu)
- **F1 Score**: 0.82-0.92
- **Training Time**: ~2-4 hours (GPU), ~20-30 hours (CPU)

### Segmentation:
- **IoU**: 0.65-0.80
- **Dice Score**: 0.70-0.85
- **Training Time**: ~3-5 hours (GPU), ~30-40 hours (CPU)

## 📁 Output Structure

```
outputs/
├── models/
│   ├── best_model.pth              # Best classification model
│   ├── best_seg_model.pth          # Best segmentation model
│   └── checkpoint_epoch_*.pth      # Training checkpoints
├── logs/
│   └── tensorboard logs
└── results/
    ├── classification_metrics.png
    ├── segmentation_metrics.png
    ├── confusion_matrix_classification.png
    ├── segmentation_evaluation.png
    └── inference_*.jpg
```

## 🔧 Troubleshooting

### 1. ImportError hoặc ModuleNotFoundError
```bash
pip install -r requirements.txt
```

### 2. CUDA Out of Memory
- Giảm `BATCH_SIZE` trong config.py
- Giảm `IMG_SIZE` xuống 224 hoặc 192
- Tắt một số augmentation

### 3. Training quá chậm
- Sử dụng GPU (CUDA)
- Giảm số epochs
- Giảm POPULATION_SIZE của SANGO
- Set `apply_gabor=False` trong preprocessing

### 4. Accuracy thấp
- Tăng số epochs
- Bật data augmentation
- Sử dụng pretrained weights (đã bật mặc định)
- Tăng learning rate hoặc dùng learning rate scheduler

## 💡 Tips để tăng hiệu suất

1. **Data Augmentation**: Đã bật mặc định, giúp tránh overfitting
2. **Mixed Precision Training**: Đã bật (AMP), giảm 50% memory usage
3. **Gradient Accumulation**: Hiệu ứng batch size lớn hơn
4. **Early Stopping**: Tự động dừng khi không improve (patience=10)
5. **Learning Rate Scheduling**: ReduceLROnPlateau cho classification

## 📝 Các lệnh hay dùng

```bash
# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test một sample nhanh
python inference.py

# Xem tensorboard logs
tensorboard --logdir outputs/logs

# Đánh giá model
python evaluate.py

# Phân tích dữ liệu
python utils.py
```

## 🎓 Hiểu về SANGO

SANGO (Self-Adaptive Northern Goshawk Optimization) được sử dụng để:
- Tự động tìm hyperparameters tối ưu cho GRU
- Cân bằng giữa exploration (tìm kiếm global) và exploitation (tối ưu local)
- Sử dụng Levy flight cho random walk hiệu quả

**Khi nào SANGO hoạt động?**
- Trong quá trình training classification model
- Tự động optimize hidden_size, dropout_rate, learning_rate

## 📊 Metrics Explained

### Classification:
- **Accuracy**: % dự đoán đúng
- **F1 Score**: Harmonic mean của precision và recall
- **Precision**: Trong những dự đoán positive, bao nhiêu % đúng
- **Recall**: Trong những case thực sự positive, model detect được bao nhiêu %

### Segmentation:
- **IoU**: Intersection over Union - độ overlap giữa prediction và ground truth
- **Dice Score**: Tương tự IoU, nhưng đánh trọng intersection cao hơn
- **Pixel Accuracy**: % pixels được phân loại đúng

## 🔬 Advanced: Custom Training

Nếu muốn customize training loop:

```python
from train_classification import ClassificationTrainer
from classification_model import create_classification_model
import config

# Create model
model = create_classification_model(num_classes=5)

# Create trainer với custom parameters
trainer = ClassificationTrainer(model, device, use_amp=True)

# Custom learning rate
trainer.optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3,  # Higher learning rate
    weight_decay=1e-4
)

# Train
trainer.train(train_loader, val_loader, num_epochs=50)
```

## 📚 References

Paper: Sharma, N., & Lalwani, P. (2025). A multi model deep net with an explainable AI based framework for diabetic retinopathy segmentation and classification. Scientific Reports, 15, 8777.

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. Run `python test_system.py` to diagnose issues
3. Check error logs in terminal
4. Reduce batch size if OOM errors occur

---

**Happy Training! 🎉**

