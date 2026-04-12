import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.config import cfg
from models.hybrid_model import Hybrid_Model
from src.dataset import ecg_arrhythmia_dataset

def plot_confusion_matrix(y_true, y_pred, classes):
    """Hàm hỗ trợ vẽ Ma trận nhầm lẫn và lưu thành file ảnh"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.ylabel('Nhãn Thực Tế (True Label)')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Ma trận nhầm lẫn đã được lưu thành 'confusion_matrix.png'")
    plt.show()

def model_evaluation(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Khởi động quá trình Đánh giá trên: {device}")

    # 1. Tải tập dữ liệu Test
    print("Đang nạp dữ liệu Test...")
    X_test_beats = np.load(f"{cfg.data_processed_path}/X_test_beats.npy")
    X_test_rr = np.load(f"{cfg.data_processed_path}/X_test_rr.npy")
    y_test = np.load(f"{cfg.data_processed_path}/y_test.npy")

    test_dataset = ecg_arrhythmia_dataset(X_test_beats, X_test_rr, y_test)

    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = Hybrid_Model(num_classes=cfg.num_classes, gru_hidden_size=cfg.gru_hidden_size).to(device)
    
    model_path = f"{cfg.models_path}/best_ecg_model.pth"
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file trọng số tại {model_path}. Hãy chạy train trước!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Đã nạp thành công trọng số từ: {model_path}")

    model.eval() 
    
    all_preds = []
    all_labels = []

    with torch.no_grad(): 
        for beats, rrs, labels in test_loader:
            beats, rrs = beats.to(device), rrs.to(device)

            outputs = model(beats, rrs)

            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*50)
    print("BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT)")
    print("="*50)

    target_names = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)', 'Q (Unknown)']
    
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)

    plot_confusion_matrix(all_labels, all_preds, target_names)