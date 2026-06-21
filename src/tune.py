import optuna
import json
import numpy as np
import os
from torch.utils.data import DataLoader
from src.config import cfg
from src.train import model_training
from src.dataset import ecg_arrhythmia_dataset

#########################################################################################################################

GLOBAL_TRAIN_LOADER = None
GLOBAL_VAL_LOADER = None

def preload_data():
    global GLOBAL_TRAIN_LOADER, GLOBAL_VAL_LOADER
    print("Đang nạp dữ liệu LÊN RAM một lần duy nhất cho Optuna...")
    
    X_train_beats = np.load(f"{cfg.data_processed_path}/X_train_beats.npy")
    X_train_rr = np.load(f"{cfg.data_processed_path}/X_train_rr.npy")
    y_train = np.load(f"{cfg.data_processed_path}/y_train.npy")
    X_val_beats = np.load(f"{cfg.data_processed_path}/X_val_beats.npy")
    X_val_rr = np.load(f"{cfg.data_processed_path}/X_val_rr.npy")
    y_val = np.load(f"{cfg.data_processed_path}/y_val.npy")

    train_dataset = ecg_arrhythmia_dataset(X_train_beats, X_train_rr, y_train)
    val_dataset = ecg_arrhythmia_dataset(X_val_beats, X_val_rr, y_val)

    return train_dataset, val_dataset

def objective(trial):
    cfg.epochs = 15
    cfg.learning_rate = trial.suggest_float("learning_rate", 8e-4, 5e-3, log=True)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    cfg.batch_size = trial.suggest_categorical("batch_size", [64, 128])
    cfg.gru_hidden_size = trial.suggest_categorical("gru_hidden_size", [32, 64, 128])
    cfg.dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.45)
    
    print(f"\nTrial {trial.number}: LR={cfg.learning_rate:.5f}, WD={cfg.weight_decay:.5f}, Batch={cfg.batch_size}, GRU={cfg.gru_hidden_size}, Drop={cfg.dropout_rate:.2f}")

    train_loader = DataLoader(GLOBAL_TRAIN_DATASET, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(GLOBAL_VAL_DATASET, batch_size=cfg.batch_size, shuffle=False)
    
    val_loss = model_training(cfg, trial=trial, ext_train_loader=train_loader, ext_val_loader=val_loader)
    return val_loss

def run_optimization(n_trials=15):
    global GLOBAL_TRAIN_DATASET, GLOBAL_VAL_DATASET
    GLOBAL_TRAIN_DATASET, GLOBAL_VAL_DATASET = preload_data()
    
    print("KHỞI ĐỘNG HỆ THỐNG OPTUNA HYPERPARAMETER TUNING...")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "="*50)
    print("QUÁ TRÌNH TUNING HOÀN TẤT!")
    print("="*50)
    print(f"Cấu hình TỐT NHẤT đạt Val Loss = {study.best_value:.4f}:")
    
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")

    os.makedirs(cfg.saved_model_path, exist_ok=True)
    param_path = os.path.join(cfg.saved_model_path, "best_optuna_params.json")
    
    with open(param_path, "w") as f:     
        json.dump(study.best_params, f, indent=4) 
        
    print(f"\nĐã lưu tự động bộ siêu tham số có độ chính xác cao vào: {param_path}")
    print("Hướng dẫn: Chỉ cần gõ lệnh train, hệ thống sẽ tự động nạp file này!")