import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import wandb

from src.config import cfg
from src.train import model_training
from src.evaluate import model_evaluation
from src.predict import run_inference

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_learning_curves(history):
    """Vẽ biểu đồ Loss và Accuracy để đưa vào báo cáo"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Biểu đồ huấn luyện đã được lưu thành 'learning_curves.png'")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="ECG Classification System")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict'])
    parser.add_argument('--patient_file', type=str, default=None)
    # <-- 2. THÊM CỜ RUN_NAME ĐỂ ĐẶT TÊN CHO LẦN CHẠY TRÊN WANDB
    parser.add_argument('--run_name', type=str, default=None, help="Tên hiển thị trên WandB (VD: cnn-bigru-lr0.001)")
    return parser.parse_args()

def main():
    set_seed(42)
    args = parse_args()
    
    print(f"Khởi động hệ thống ECG. Chế độ: {args.mode.upper()}")
    
    if args.mode == 'train':
        # <-- 3. KHỞI TẠO WANDB TRƯỚC KHI TRAIN
        wandb.init(
            project="ecg-arrhythmia-detection", # Tên dự án (Sẽ tạo 1 folder trên WandB)
            name=args.run_name if args.run_name else "Hybrid-CNN-BiGRU", # Tên lần chạy
            config=cfg.__dict__ if hasattr(cfg, '__dict__') else cfg # Lưu toàn bộ file config lên mây!
        )
        
        model, history = model_training(cfg)
        plot_learning_curves(history)
        
        # <-- 4. KẾT THÚC PHIÊN WANDB
        wandb.finish()
        
    elif args.mode == 'test':
        print("Đang đánh giá mô hình trên tập Test...")
        model_evaluation(cfg)
        
    elif args.mode == 'predict':
        if not args.patient_file:
            print("Lỗi: Cần cung cấp --patient_file cho chế độ predict.")
            return
        print(f"Đang chẩn đoán cho dữ liệu: {args.patient_file}")
        run_inference(cfg, args.patient_file)

if __name__ == "__main__":
    main()