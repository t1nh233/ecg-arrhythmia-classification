import argparse
import torch
import numpy as np
import random
import os
import json
from src.config import cfg
from src.train import model_training
from src.predict import run_inference

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="ECG Classification System")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict', 'tune'])
    parser.add_argument('--patient_file', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None, help="Tên hiển thị trên WandB (VD: cnn-bigru-lr0.001)")
    return parser.parse_args()

def main():
    set_seed(42)
    args = parse_args()
    
    print(f"Khởi động hệ thống ECG. Chế độ: {args.mode.upper()}")

    if args.mode == 'train':
        param_path = os.path.join(cfg.saved_model_path, "best_optuna_params.json")
        if os.path.exists(param_path):
            print(f"\nPhát hiện file cấu hình Optuna ({param_path}).")
            print("Đang tiến hành ghi đè config gốc để đạt độ chính xác tối đa...")
            
            with open(param_path, "r") as f:
                best_params = json.load(f)
                
            for key, value in best_params.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value) # Cập nhật động vào object cfg
                    print(f"   + Đã cập nhật {key}: {value}")

        else:
            print("\nℹKhông tìm thấy file cấu hình Optuna, sử dụng cấu hình mặc định trong config.py")
        print("-" * 50)


        import wandb
        wandb.init(
            project="ecg-arrhythmia-detection",
            name=args.run_name if args.run_name else "Hybrid-CNN-BiGRU",
            config=cfg.__dict__ if hasattr(cfg, '__dict__') else cfg
        )

        model, history = model_training(cfg)
        
        wandb.finish()
        
    elif args.mode == 'predict':
        if not args.patient_file:
            print("Lỗi: Cần cung cấp --patient_file cho chế độ predict.")
            return
        print(f"🩺 Đang chẩn đoán cho dữ liệu: {args.patient_file}")
        run_inference(cfg, args.patient_file)

    elif args.mode == 'tune':
        from src.tune import run_optimization
        run_optimization(n_trials=15)

if __name__ == "__main__":
    main()