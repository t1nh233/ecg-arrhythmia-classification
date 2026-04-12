import numpy as np 

#########################################################################################################################

def segment_norm_signal(signal, valid_r, valid_r_idx, rr_features, eps=1e-8):
    beats = []
    valid_idx = []
    valid_rr = []

    for i, (idx, r) in enumerate(zip(valid_r_idx, valid_r)):
        if r - 100 < 0 or r + 200 > len(signal):
            continue
        beats.append(signal[r - 100 : r + 200])
        valid_idx.append(idx)
        
        # Gọi rr_features theo biến đếm 'i' (0, 1, 2...)
        valid_rr.append(rr_features[i]) 

    beats_array = np.array(beats)
    mean = np.mean(beats_array, axis=1, keepdims=True)
    std = np.std(beats_array, axis=1, keepdims=True)
    std = np.maximum(std, eps)

    beat_normalized = (beats_array - mean) / std

    return beat_normalized, np.array(valid_idx), np.array(valid_rr)