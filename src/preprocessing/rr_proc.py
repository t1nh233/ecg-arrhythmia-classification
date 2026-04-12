import numpy as np

#########################################################################################################################

def extract_rr_feature(r_index, fs=360):

    all_rr_interval = np.diff(r_index)

    rr_feature = []
    valid_r = []
    valid_idx = []

    for i in range(5, len(r_index) - 5):
        rr_pre = all_rr_interval[i - 1] / fs
        rr_post = all_rr_interval[i] / fs
        rr_avg = np.mean(all_rr_interval[i - 5: i + 5]) / fs

        rr_feature.append([rr_pre, rr_post, rr_avg])

        valid_r.append(r_index[i])
        valid_idx.append(i)

    return np.array(rr_feature), np.array(valid_r), np.array(valid_idx)