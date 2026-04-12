from dataclasses import dataclass

@dataclass
class ECGConfig:

    data_raw_path: str = "./dataset/raw"
    data_processed_path: str = "./dataset/processed"
    models_path: str = "./models/saved_models"
    device: str = "cuda"

    num_classes: int = 5
    signal_len: int = 300

    gru_hidden_size: int = 32
    dropout_rate: float = 0.2

    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

cfg = ECGConfig()