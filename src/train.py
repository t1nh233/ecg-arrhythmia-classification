import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import copy
from src.models.hybrid_model import Hybrid_Model
from src.dataset import ecg_arrhythmia_dataset
from src.evaluate import Model_Evaluator

#########################################################################################################################

def model_training(cfg, trial=None, ext_train_loader=None, ext_val_loader=None):

    os.makedirs(cfg.saved_model_path, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if trial is None:
        print(f"Huấn luyện trên thiết bị: {device}")

    if ext_train_loader is not None and ext_val_loader is not None:
        train_loader = ext_train_loader
        val_loader = ext_val_loader

    else:
        print("Đang tiến hành nạp dữ liệu...")
        X_train_beats = np.load(f"{cfg.data_processed_path}/X_train_beats.npy")
        X_train_rr = np.load(f"{cfg.data_processed_path}/X_train_rr.npy")
        y_train = np.load(f"{cfg.data_processed_path}/y_train.npy")

        X_val_beats = np.load(f"{cfg.data_processed_path}/X_val_beats.npy")
        X_val_rr = np.load(f"{cfg.data_processed_path}/X_val_rr.npy")
        y_val = np.load(f"{cfg.data_processed_path}/y_val.npy")

        train_dataset = ecg_arrhythmia_dataset(X_train_beats, X_train_rr, y_train)
        val_dataset = ecg_arrhythmia_dataset(X_val_beats, X_val_rr, y_val)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = Hybrid_Model(num_classes=cfg.num_classes, gru_hidden_size=cfg.gru_hidden_size).to(device)

    ## Cross entropy loss voi weight (1.0, 5.0, 2.0)
    class_weights = torch.tensor([1.0, 5.0, 2.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 6
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("Bắt đầu quá trình huấn luyện...")
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for beats, rrs, labels in train_loader:
            beats, rrs, labels = beats.to(device), rrs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(beats, rrs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for beats, rrs, labels in val_loader:
                beats, rrs, labels = beats.to(device), rrs.to(device), labels.to(device)
                outputs = model(beats, rrs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if trial is None:
            print(f"Epoch [{epoch+1:02d}/{cfg.epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if trial is not None:
            import optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_model_weights, f"{cfg.saved_model_path}/best_ecg_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Dừng sớm tại Epoch {epoch+1}. Không cải thiện trong {patience} epochs.")
                break
    
    if trial is not None: 
        return best_val_loss

    print("\n" + "*"*50)
    print("HOÀN TẤT HUẤN LUYỆN. TIẾN HÀNH ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT...")
    print("*"*50)
 
    model.load_state_dict(best_model_weights)
    model.eval()
    
    print("Đang nạp tập Test để đánh giá khách quan...")
    X_test_beats = np.load(f"{cfg.data_processed_path}/X_test_beats.npy")
    X_test_rr = np.load(f"{cfg.data_processed_path}/X_test_rr.npy")
    y_test = np.load(f"{cfg.data_processed_path}/y_test.npy")
    test_dataset = ecg_arrhythmia_dataset(X_test_beats, X_test_rr, y_test)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for beats, rrs, labels in test_loader:
            beats, rrs = beats.to(device), rrs.to(device)
            outputs = model(beats, rrs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    target_names = ['N', 'S', 'V']
    evaluator = Model_Evaluator(class_names=target_names)
    
    evaluator.plot_learning_curves(history, cfg.fig_path)
    evaluator.plot_confusion_matrix(all_labels, all_preds, cfg.fig_path)
    evaluator.generate_quantitative_report(all_labels, all_preds, cfg.report_path)

    return model, history

