import os
import sys
# Thêm thư mục gốc dự án vào sys.path để tránh lỗi ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import cfg
from src.models.hybrid_model import Hybrid_Model
from src.constant import IDX_TO_CLASS
from src.p_preprocessing.proc_patient import process_patient, load_patient_data

app = FastAPI(title="ECG Arrhythmia Classification System", version="1.0.0")

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Frontend static files not found yet. Please create static/index.html</h2>")

@app.get("/api/download/sample/ekg")
def download_sample_ekg():
    file_path = os.path.join(cfg.data_raw_path, "100_ekg.csv")
    if not os.path.exists(file_path):
        # Fallback to scanning for any ekg file
        files = glob.glob(os.path.join(cfg.data_raw_path, "*_ekg.csv"))
        if not files:
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp EKG mẫu nào.")
        file_path = files[0]
        
    try:
        df = pd.read_csv(file_path, nrows=1000)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "sample_ekg.csv")
        df.to_csv(temp_path, index=False)
        return FileResponse(temp_path, media_type="text/csv", filename="sample_ekg.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo tệp mẫu: {str(e)}")

@app.get("/api/download/sample/annotation")
def download_sample_annotation():
    file_path = os.path.join(cfg.data_raw_path, "100_annotations_1.csv")
    if not os.path.exists(file_path):
        files = glob.glob(os.path.join(cfg.data_raw_path, "*_annotations_1.csv"))
        if not files:
            raise HTTPException(status_code=404, detail="Không tìm thấy tệp nhãn mẫu nào.")
        file_path = files[0]
        
    try:
        df = pd.read_csv(file_path, nrows=100)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "sample_annotations_1.csv")
        df.to_csv(temp_path, index=False)
        return FileResponse(temp_path, media_type="text/csv", filename="sample_annotations_1.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo tệp mẫu: {str(e)}")

@app.get("/api/patients")
def list_patients():
    raw_path = cfg.data_raw_path
    if not os.path.exists(raw_path):
        return []
    
    # List all files matching *_ekg.csv
    files = glob.glob(os.path.join(raw_path, "*_ekg.csv"))
    patient_ids = []
    for f in files:
        basename = os.path.basename(f)
        pid = basename.split("_ekg.csv")[0]
        # Check if corresponding annotations file exists
        ann_path = os.path.join(raw_path, f"{pid}_annotations_1.csv")
        if os.path.exists(ann_path):
            patient_ids.append(pid)
            
    return sorted(list(set(patient_ids)))

@app.post("/api/predict/patient/{patient_id}")
def predict_patient(patient_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ekg_path = os.path.join(cfg.data_raw_path, f"{patient_id}_ekg.csv")
    ann_path = os.path.join(cfg.data_raw_path, f"{patient_id}_annotations_1.csv")
    
    if not os.path.exists(ekg_path) or not os.path.exists(ann_path):
        raise HTTPException(status_code=404, detail=f"Dữ liệu của bệnh nhân {patient_id} không đầy đủ hoặc không tìm thấy.")
        
    try:
        signal, r_index, labels = load_patient_data(patient_id, cfg.data_raw_path)
        beats, valid_rr_seg, mapped_labels = process_patient(signal, r_index, labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tiền xử lý tín hiệu: {str(e)}")
        
    if len(beats) == 0:
        raise HTTPException(status_code=400, detail="Không thể trích xuất nhịp tim hợp lệ từ dữ liệu bệnh nhân.")

    # Load Model
    model = Hybrid_Model(num_classes=cfg.num_classes, gru_hidden_size=cfg.gru_hidden_size).to(device)
    model_path = os.path.join(cfg.saved_model_path, "best_ecg_model.pth")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Không tìm thấy trọng số model AI (best_ecg_model.pth).")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Process
    beats_tensor = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(device)
    rrs_tensor = torch.tensor(valid_rr_seg, dtype=torch.float32).to(device)
    
    full_names = {
        'N': 'Bình thường',
        'S': 'Ngoại tâm thu trên thất (S)',
        'V': 'Ngoại tâm thu thất (V)'
    }
    
    results = []
    with torch.no_grad():
        outputs = model(beats_tensor, rrs_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_probs, predicted_indices = torch.max(probabilities, 1)
        
        for i in range(len(predicted_indices)):
            pred_idx = predicted_indices[i].item()
            confidence = max_probs[i].item()
            ai_label = IDX_TO_CLASS[pred_idx]
            
            true_label = IDX_TO_CLASS[int(mapped_labels[i])] if i < len(mapped_labels) else "N/A"
            is_correct = bool(ai_label == true_label)
            
            results.append({
                "beat_index": i + 1,
                "prediction": full_names.get(ai_label, ai_label),
                "confidence": float(confidence),
                "ground_truth": full_names.get(true_label, "Không xác định"),
                "is_correct": is_correct,
                "signal": beats[i].tolist()  # 300 points
            })
            
    total_beats = len(results)
    correct_beats = sum(1 for r in results if r["is_correct"])
    accuracy = (correct_beats / total_beats) if total_beats > 0 else 0.0
    
    class_counts = {"N": 0, "S": 0, "V": 0}
    for r in results:
        abbr = "N" if "Bình thường" in r["prediction"] else ("S" if "(S)" in r["prediction"] else "V")
        class_counts[abbr] += 1
        
    return {
        "patient_id": patient_id,
        "accuracy": accuracy,
        "total_beats": total_beats,
        "class_counts": class_counts,
        "results": results[:100]  # Cap at 100 beats for frontend rendering efficiency
    }

from scipy.signal import find_peaks
from src.p_preprocessing.filter_signal import preprocess_signal
from src.p_preprocessing.rr_proc import extract_rr_feature
from src.p_preprocessing.segment_beat import segment_norm_signal

def detect_r_peaks(signal, fs=360):
    # R-peaks are usually the sharpest positive deflections
    # We find peaks with a minimum distance of 200ms (72 samples at 360Hz)
    # The height threshold is set dynamically based on signal percentile
    height_thresh = float(np.percentile(signal, 75))
    peaks, _ = find_peaks(signal, distance=72, height=height_thresh)
    return peaks

@app.post("/api/predict/upload")
def predict_upload(ekg_file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Save files to temp directory
    temp_dir = tempfile.mkdtemp()
    ekg_path = os.path.join(temp_dir, "uploaded_ekg.csv")
    
    try:
        with open(ekg_path, "wb") as buffer:
            shutil.copyfileobj(ekg_file.file, buffer)
            
        df = pd.read_csv(ekg_path)
        if 'MLII' in df.columns:
            signal = df['MLII'].values
        else:
            signal = df.iloc[:, 0].values
            
        # 1. Denoise and clean signal
        clean_signal = preprocess_signal(signal, fs=360)
        
        # 2. Automatically locate R-peaks
        r_peaks = detect_r_peaks(clean_signal, fs=360)
        
        if len(r_peaks) < 10:
            # Fallback to lower threshold if not enough peaks detected
            height_thresh = float(np.percentile(clean_signal, 50))
            r_peaks, _ = find_peaks(clean_signal, distance=72, height=height_thresh)
            
        if len(r_peaks) < 10:
            raise ValueError("Không tìm thấy đủ số lượng nhịp tim (R-peaks) trong tệp tin.")
            
        # 3. Extract RR Features
        rr_features, valid_r, valid_idx = extract_rr_feature(r_peaks, fs=360)
        
        # 4. Segment heartbeats
        beats, _, final_rr = segment_norm_signal(clean_signal, valid_r, valid_idx, rr_features)
        
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file tải lên: {str(e)}")
        
    shutil.rmtree(temp_dir, ignore_errors=True)
        
    if len(beats) == 0:
        raise HTTPException(status_code=400, detail="Không thể trích xuất nhịp tim hợp lệ từ file tải lên.")
        
    # Load Model
    model = Hybrid_Model(num_classes=cfg.num_classes, gru_hidden_size=cfg.gru_hidden_size).to(device)
    model_path = os.path.join(cfg.saved_model_path, "best_ecg_model.pth")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Không tìm thấy trọng số model AI.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    beats_tensor = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(device)
    rrs_tensor = torch.tensor(final_rr, dtype=torch.float32).to(device)
    
    full_names = {
        'N': 'Bình thường',
        'S': 'Ngoại tâm thu trên thất (S)',
        'V': 'Ngoại tâm thu thất (V)'
    }
    
    results = []
    with torch.no_grad():
        outputs = model(beats_tensor, rrs_tensor)
        probabilities = F.softmax(outputs, dim=1)
        max_probs, predicted_indices = torch.max(probabilities, 1)
        
        for i in range(len(predicted_indices)):
            pred_idx = predicted_indices[i].item()
            confidence = max_probs[i].item()
            ai_label = IDX_TO_CLASS[pred_idx]
            
            results.append({
                "beat_index": i + 1,
                "prediction": full_names.get(ai_label, ai_label),
                "confidence": float(confidence),
                "ground_truth": "Không xác định (AI Tự động phát hiện)",
                "is_correct": True, # Automatically true since no doctor labels uploaded
                "signal": beats[i].tolist()
            })
            
    total_beats = len(results)
    
    class_counts = {"N": 0, "S": 0, "V": 0}
    for r in results:
        abbr = "N" if "Bình thường" in r["prediction"] else ("S" if "(S)" in r["prediction"] else "V")
        class_counts[abbr] += 1
        
    return {
        "patient_id": "Tệp tự động phát hiện",
        "accuracy": 1.0,
        "total_beats": total_beats,
        "class_counts": class_counts,
        "results": results[:100]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
