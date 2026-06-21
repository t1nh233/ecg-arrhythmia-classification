# file: src/predict.py
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd

from src.config import cfg
from src.models.hybrid_model import Hybrid_Model
from src.constant import CLASS_TO_IDX, IDX_TO_CLASS
from src.p_preprocessing.proc_patient import build_dataset

#########################################################################################################################

def preprocess_patient_data(patient_input, data_raw_path):
    print(f"Đang tiền xử lý tín hiệu cho bệnh nhân: {patient_input}...")

    patient_id = str(patient_input).replace('\\', '/').split('/')[-1].split('.')[0]

    patient_ids = [patient_id]

    X_beats, X_rr, y_true = build_dataset(patient_ids, data_raw_path)
    
    if len(X_beats) == 0:
        raise ValueError(f"Không tìm thấy nhịp tim nào hợp lệ cho bệnh nhân {patient_id}!")
        
    print(f"Trích xuất thành công {len(X_beats)} nhịp tim.")

    beats_tensor = torch.tensor(X_beats, dtype=torch.float32).unsqueeze(1) 
    rrs_tensor = torch.tensor(X_rr, dtype=torch.float32)
    
    return beats_tensor, rrs_tensor, y_true, patient_id

def run_inference(cfg, patient_file):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    try:
        beats, rrs, y_true, clean_patient_id = preprocess_patient_data(patient_file, cfg.data_raw_path)
        beats = beats.to(device)
        rrs = rrs.to(device)
    except Exception as e:
        print(e)
        return
    
    # 2. Tải Mô hình & Trọng số
    model = Hybrid_Model(num_classes=cfg.num_classes, gru_hidden_size=cfg.gru_hidden_size).to(device)
    model_path = f"{cfg.saved_model_path}/best_ecg_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy trọng số tại {model_path}!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Đã nạp Model AI thành công. Bắt đầu chẩn đoán...\n")
    
    # 3. Suy luận (Inference)
    model.eval()
    results = []

    full_names = {
        'N': 'Bình thường',
        'S': 'Ngoại tâm thu trên thất',
        'V': 'Ngoại tâm thu thất'
    }

    with torch.no_grad():
        outputs = model(beats, rrs)
        probabilities = F.softmax(outputs, dim=1)
        max_probs, predicted_indices = torch.max(probabilities, 1)
        
        # 4. Gom dữ liệu tạo Báo cáo
        for i in range(len(predicted_indices)):
            pred_idx = predicted_indices[i].item()
            confidence = max_probs[i].item() * 100
            
            ai_label = IDX_TO_CLASS[pred_idx]
            
            # Xử lý nhãn thực tế (Ground Truth) nếu có
            if y_true is not None and len(y_true) > i:
                true_idx = int(y_true[i])
                doctor_label = IDX_TO_CLASS[true_idx]
                is_correct = "Đúng" if ai_label == doctor_label else "Sai"
            else:
                doctor_label = "N/A"
                is_correct = "-"

            results.append({
                "Nhịp số": i + 1,
                "AI Dự Đoán": full_names[ai_label],
                "Độ tin cậy": f"{confidence:.2f}%",
                "Thực tế": full_names.get(doctor_label, "N/A"),
                "Đánh giá": is_correct
            })

    # 5. In bảng kết quả cực kỳ chuyên nghiệp
    print("="*70)
    print(f" BÁO CÁO CHẨN ĐOÁN BỆNH NHÂN: {clean_patient_id} (Tổng số nhịp: {len(beats)})")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    
    # Mẹo in Pandas DataFrame căn lề đẹp trên Terminal
    pd.set_option('display.max_rows', 50) 
    print(df_results.to_string(index=False))
    
    # Tính tỷ lệ chính xác trên file này (Ngoại trừ các nhịp không có nhãn)
    if y_true is not None:
        correct_count = sum(1 for r in results if "Đúng" in r["Đánh giá"])
        acc = (correct_count / len(results)) * 100
        print("-" * 70)
        print(f"Độ chính xác của AI trên bệnh nhân {clean_patient_id}: {acc:.2f}%")
        print("-" * 70)

    # Lưu file
    output_csv = f"{cfg.data_processed_path}/patient_{clean_patient_id}_report.csv"
    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu báo cáo y khoa thành công vào: {output_csv}")