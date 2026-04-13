import numpy as np
from src.constant import VALID_SYMBOLS, AAMI_MAP, CLASS_TO_IDX

#########################################################################################################################

def filter_valid_beats(r_index, labels):

    clean_r = []
    clean_labels = []

    for r, l in zip(r_index, labels):
        if l in VALID_SYMBOLS:
            clean_r.append(r)
            clean_labels.append(l)

    return np.array(clean_r), np.array(clean_labels)

def map_to_aami(labels):

    return np.array([CLASS_TO_IDX[AAMI_MAP[l]] for l in labels])