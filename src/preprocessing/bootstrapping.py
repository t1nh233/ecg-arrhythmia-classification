import numpy as np
from sklearn.utils import resample

def apply_bootstrapping(X_beats, X_rr, y, target_samples=30000, random_state=42):

    print(f"Bắt đầu Bootstrapping ép mỗi lớp về {target_samples} mẫu...")
    classes = np.unique(y)
    
    X_beats_bal, X_rr_bal, y_bal = [], [], []
    
    for cls in classes:
        # Lấy index của lớp hiện tại
        idx_cls = np.where(y == cls)[0]
        
        # Quyết định: Resample có hoàn lại (True) nếu mẫu ít hơn target, ngược lại False
        replace_flag = len(idx_cls) < target_samples 
        
        beats_res, rr_res, y_res = resample(
            X_beats[idx_cls], 
            X_rr[idx_cls], 
            y[idx_cls],
            replace=replace_flag,
            n_samples=target_samples,
            random_state=random_state
        )
        
        X_beats_bal.append(beats_res)
        X_rr_bal.append(rr_res)
        y_bal.append(y_res)
        
        print(f" - Lớp {cls}: Từ {len(idx_cls)} -> {target_samples} mẫu.")
        
    return np.vstack(X_beats_bal), np.vstack(X_rr_bal), np.concatenate(y_bal)