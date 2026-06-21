import numpy as np
from src.p_preprocessing.proc_patient import build_dataset, split_patients
from src.p_preprocessing.bootstrapping import apply_bootstrapping
import os
from src.config import cfg
from src.constant import PATIENT_IDS

#########################################################################################################################

def main():

    os.makedirs(cfg.data_processed_path, exist_ok=True)

    print("1. Đang chia tập Train/Val/Test theo Patient ID...")
    train_val_ids, test_ids = split_patients(PATIENT_IDS, test_size=0.2)
    train_ids, val_ids = split_patients(train_val_ids, test_size=0.2)
    print(f"Số bệnh nhân Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    print("\n2. Đang xử lý tập Train...")
    X_train_raw, X_train_rr_raw, y_train_raw = build_dataset(train_ids, cfg.data_raw_path)

    print("\n3. Đang xử lý tập Val...")
    X_val_beats, X_val_rr, y_val = build_dataset(val_ids, cfg.data_raw_path)

    print("\n4. Đang xử lý tập Test...")
    X_test_beats, X_test_rr, y_test = build_dataset(test_ids, cfg.data_raw_path)
    
    print("\n5. Áp dụng Bootstrapping cho tập Train...")
    X_train_bal, X_train_rr_bal, y_train_bal = apply_bootstrapping(X_train_raw, X_train_rr_raw, y_train_raw, target_samples=30000)

    print("\n6. Lưu dữ liệu đã xử lý thành công...")
    np.save(f"{cfg.data_processed_path}/X_train_beats.npy", X_train_bal)
    np.save(f"{cfg.data_processed_path}/X_train_rr.npy", X_train_rr_bal)
    np.save(f"{cfg.data_processed_path}/y_train.npy", y_train_bal)
    
    np.save(f"{cfg.data_processed_path}/X_val_beats.npy", X_val_beats)
    np.save(f"{cfg.data_processed_path}/X_val_rr.npy", X_val_rr)
    np.save(f"{cfg.data_processed_path}/y_val.npy", y_val)

    np.save(f"{cfg.data_processed_path}/X_test_beats.npy", X_test_beats)
    np.save(f"{cfg.data_processed_path}/X_test_rr.npy", X_test_rr)
    np.save(f"{cfg.data_processed_path}/y_test.npy", y_test)

    print("\nQuá trình tiền xử lý dữ liệu đã thành công !!")

if __name__ == "__main__":
    main()