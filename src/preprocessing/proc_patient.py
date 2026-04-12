from preprocessing.filter_signal import preprocess_signal
from preprocessing.mapping_ann import filter_valid_beats, map_to_aami
from preprocessing.rr_proc import extract_rr_feature
from preprocessing.segment_beat import segment_norm_signal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#########################################################################################################################

def load_patient_data(patient_id, data_dir):

    signal_path = f"{data_dir}/{patient_id}_ekg.csv"
    ann_path = f"{data_dir}/{patient_id}_annotations_1.csv"

    signal = pd.read_csv(signal_path).values.squeeze()

    ann = pd.read_csv(ann_path)
    r_index = ann['sample'].values
    labels = ann['symbol'].values

    return signal, r_index, labels


def split_patients(patient_ids, test_size=0.2, random_state=42):

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state
    )

    return train_ids, test_ids


def process_patient(signal, r_index, labels):

    signal_filtered = preprocess_signal(signal)

    r_index, labels = filter_valid_beats(r_index, labels)
    rr, valid_r_exrr, valid_idx_exrr = extract_rr_feature(r_index)
    beats, valid_idx_seg, valid_rr_seg = segment_norm_signal(signal_filtered, valid_r_exrr, valid_idx_exrr, rr)

    labels = labels[valid_idx_seg]
    labels = map_to_aami(labels)

    return beats, valid_rr_seg, labels


def build_dataset(patient_ids, data_dir):

    all_beats, all_rr, all_labels = [], [], []

    for pid in patient_ids:
        signal, r_index, labels = load_patient_data(pid, data_dir)

        beats, rr, y = process_patient(signal, r_index, labels)

        all_beats.append(beats)
        all_rr.append(rr)
        all_labels.append(y)

    X = np.concatenate(all_beats)
    RR = np.concatenate(all_rr)
    y = np.concatenate(all_labels)

    return X, RR, y


