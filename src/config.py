from dataclasses import dataclass

#########################################################################################################################

@dataclass
class ECGConfig:

    data_raw_path: str = "./dataraw/raw"
    data_processed_path: str = "./data/processed"
    saved_model_path: str = "./checkpoints"
    fig_path: str = "./outputs/figure"
    report_path: str = "./outputs/report"
    device: str = "cuda"

    num_classes: int = 3
    signal_len: int = 300

    gru_hidden_size: int = 32
    gru_num_layer: int = 1
    dropout_rate: float = 0.2

    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

cfg = ECGConfig()