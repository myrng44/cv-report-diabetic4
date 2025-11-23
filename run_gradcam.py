"""
Script nhanh để chạy trực quan hóa Grad-CAM trên mô hình đã huấn luyện
Cách dùng: python run_gradcam.py
"""

import torch
import os
from gradcam_visualization import visualize_predictions_with_gradcam, batch_visualize_gradcam
from classification_model import create_classification_model

def main():
    print("\n" + "="*80)
    print("🔍 TRỰC QUAN HÓA GRAD-CAM CHO PHÂN LOẠI DR")
    print("="*80 + "\n")

    # Thiết lập
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}\n")

    # Tải mô hình
    print("Đang tải mô hình đã huấn luyện...")
    model = create_classification_model(num_classes=5, pretrained=False)

    # Thử tải mô hình phân loại
    model_paths = [
        'outputs/models/best_classification_model.pth',
        'outputs/models/best_model_ultra.pth',
        'outputs/models/best_model.pth'
    ]

    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Xử lý cả state_dict trực tiếp và dict checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from: {model_path}\n")
            model_loaded = True
            break

    if not model_loaded:
        print("✗ Không tìm thấy mô hình đã huấn luyện!")
        print("Vui lòng huấn luyện mô hình trước bằng cách dùng:")
        print("  python train_ultra_optimized.py")
        print("  or")
        print("  python main.py --mode train_classification")
        return

    model = model.to(device)
    model.eval()

    # Tạo thư mục đầu ra
    os.makedirs('outputs/results/gradcam_visualizations', exist_ok=True)

    # Tùy chọn 1: Trực quan hóa ảnh cụ thể
    test_image = 'data/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_01.jpg'

    if os.path.exists(test_image):
        print("Generating Grad-CAM for sample image...")
        visualize_predictions_with_gradcam(
            model,
            test_image,
            device,
            save_path='outputs/results/gradcam_example.png'
        )
        print()

    # Tùy chọn 2: Trực quan hóa hàng loạt (10 mẫu ngẫu nhiên)
    train_dir = 'data/B. Disease Grading/1. Original Images/a. Training Set'

    if os.path.exists(train_dir):
        print(f"Generating Grad-CAM for 10 random training images...")
        batch_visualize_gradcam(
            model,
            train_dir,
            device,
            'outputs/results/gradcam_visualizations',
            num_samples=10
        )

    print("\n" + "="*80)
    print("✅ TRỰC QUAN HÓA GRAD-CAM HOÀN THÀNH!")
    print("="*80)
    print("\nKiểm tra kết quả tại:")
    print("  - outputs/results/gradcam_example.png")
    print("  - outputs/results/gradcam_visualizations/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
