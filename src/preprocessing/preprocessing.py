import numpy as np
from preprocessing.proc_patient import build_dataset, split_patients
from preprocessing.bootstrapping import apply_bootstrapping
import os
from src.config import cfg
from constant import PATIENT_IDS

def main():
    os.makedirs(cfg.data_processed_path, exist_ok=True)

    print("1. Đang chia tập Train/Test theo mã bệnh nhân...")
    train_ids, test_ids = split_patients(PATIENT_IDS, test_size=0.2)
    print(f"Số bệnh nhân Train: {len(train_ids)} | Test: {len(test_ids)}")

    print("\n2. Đang xử lý tập TEST (Có thể mất vài phút)...")
    X_test_beats, X_test_rr, y_test = build_dataset(test_ids, cfg.data_raw_path)

    print("\n3. Đang xử lý tập TRAIN (Có thể mất vài phút)...")
    X_train_raw, X_train_rr_raw, y_train_raw = build_dataset(train_ids, cfg.data_raw_path)
    
    print("\n4. Áp dụng Bootstrapping cho tập TRAIN...")
    X_train_bal, X_train_rr_bal, y_train_bal = apply_bootstrapping(X_train_raw, X_train_rr_raw, y_train_raw, target_samples=30000)

    print("\n5. Lưu dữ liệu đã hoàn thiện...")
    np.save(f"{cfg.data_processed_path}/X_train_beats.npy", X_train_bal)
    np.save(f"{cfg.data_processed_path}/X_train_rr.npy", X_train_rr_bal)
    np.save(f"{cfg.data_processed_path}/y_train.npy", y_train_bal)
    
    np.save(f"{cfg.data_processed_path}/X_test_beats.npy", X_test_beats)
    np.save(f"{cfg.data_processed_path}/X_test_rr.npy", X_test_rr)
    np.save(f"{cfg.data_processed_path}/y_test.npy", y_test)

    print("\nDATA PIPELINE HOÀN TẤT MỸ MÃN! Dữ liệu đã sẵn sàng cho Deep Learning.")

if __name__ == "__main__":
    main()