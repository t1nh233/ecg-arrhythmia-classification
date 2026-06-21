# ECG Arrhythmia Classification System

An end-to-end deep learning system for detecting cardiac arrhythmias from ECG (Electrocardiogram) signals. The project uses a hybrid **CNN-BiGRU** model structure to classify single ECG beats into AAMI categories: **N** (Normal), **S** (Supraventricular ectopic), and **V** (Ventricular ectopic).

---

## 🚀 Key Features

* **Advanced Deep Learning Model**: Combines Convolutional Neural Networks (CNN) for spatial feature extraction and Bidirectional Gated Recurrent Units (BiGRU) for temporal pattern modeling of heartbeat intervals.
* **Leakage-Free Splitting**: Implements strict patient-level dataset partitioning (Train: 28 patients, Val: 7 patients, Test: 9 patients) to prevent data leakage during optimization.
* **Optuna Tuning & WandB Support**: Automatically searches hyperparameters using Optuna and tracks logs and models using Weights & Biases (WandB).
* **Self-Balancing (Bootstrapping)**: Corrects severe class imbalance by resampling minority heartbeat categories up to 30,000 samples.
* **Web Dashboard & API**: Interactive diagnosis dashboard using FastAPI (Backend) and a sleek glassmorphic UI (Frontend) with real-time heartbeat visualizations.
* **Fully Dockerized**: Built-in Docker configuration for quick local deployment.

---

## 📊 Model Evaluation Results (Test Set)

Evaluated strictly on the **unseen Test Set** (completely separate patient cohort):

* **Overall Test Accuracy**: **88.36%**
* **Class N (Normal)**: Sensitivity **90.61%**, Specificity **78.90%**, F-score **94.00%**
* **Class S (Supraventricular)**: Sensitivity **35.05%**, Specificity **98.47%**, F-score **35.56%**
* **Class V (Ventricular)**: Sensitivity **77.60%**, Specificity **91.23%**, F-score **52.74%**

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Environment
```bash
git clone <repository-url>
cd ecg-arrhythmia-classification

# Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Data
Extract the raw MIT-BIH database (CSV format) into:
```
dataraw/raw/
  ├── 100_ekg.csv
  ├── 100_annotations_1.csv
  └── ...
```

---

## 💻 Running the Pipeline

### Step 1: Preprocessing
Processes raw signals, segments individual heartbeats, extracts RR intervals, applies class bootstrapping, and saves the data splits inside `data/processed/`.
```bash
python main.py --mode train  # If running from scratch or:
# Preprocessing logic executes automatically at start of data preparation
```

### Step 2: Hyperparameter Tuning (Optional)
Utilizes Optuna to tune model parameters:
```bash
python main.py --mode tune
```

### Step 3: Model Training
Trains the hybrid network. It will automatically load the optimal Optuna config (if found in `checkpoints/best_optuna_params.json`).
```bash
python main.py --mode train --run_name "cnn-bigru-v1"
```

### Step 4: Local CLI Inference
Diagnose a single patient's file:
```bash
python main.py --mode predict --patient_file "dataraw/raw/100"
```

---

## ☁️ Google Colab Training
For training on cloud GPUs (e.g., Google Colab):
1. Zip the workspace directory excluding `venv/` and `dataraw/` to a file named `ECG_Ready_To_Train.zip`.
2. Upload `ECG_Ready_To_Train.zip` to Google Drive at: `MyDrive/Colab_Workspace/ECG_Ready_To_Train.zip`.
3. Open and run the Jupyter notebook located at `notebooks/train_notebook.ipynb` in Google Colaboratory.

---

## 🌐 Web Dashboard & API

### Run Locally
Start the FastAPI server:
```bash
python src/app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Features
* **Interactive Diagnosis**: Choose from the dropdown list of available patients to instantly perform an AI-driven ECG analysis.
* **Signal Plotter**: View individual heartbeat signals overlayed with classification labels.
* **Custom File Upload**: Upload custom EKG and Annotation CSVs for live diagnostics.

---

## 🐳 Docker Deployment

To build and launch the application in a secure Docker container:

```bash
# Build the Docker image
docker build -t ecg-arrhythmia-classification .

# Run the container on port 8000
docker run -p 8000:8000 ecg-arrhythmia-classification
```
Visit **[http://localhost:8000](http://localhost:8000)** to interact with the dashboard.
