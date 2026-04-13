import numpy as np
from scipy.signal import cheby2, sosfiltfilt
import pywt

#########################################################################################################################

def cheby2_bandpass_filter(signal, fs=360, lowcut=0.5, highcut=48, order=4, rs=40):

  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq

  sos = cheby2(order, rs, [low, high], btype='bandpass', output='sos')

  padlen = min(len(signal) - 1, 100)
  filtered = sosfiltfilt(sos, signal, padlen=padlen)

  return filtered


def wavelet_denoise(signal, wavelet='db4', level=6):

  try:
    coeffs = pywt.wavedec(signal, wavelet, level=level)

  except ValueError:
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    level = min(level, max_level)
    coeffs = pywt.wavedec(signal, wavelet, level=level)

  cA = coeffs[0]
  cDs = coeffs[1:]

  sigma = np.median(np.abs(cDs[-1])) / 0.6745
  base_thresh = sigma * np.sqrt(2 * np.log(len(signal))) * 0.8
  num_levels = len(cDs)

  new_cDs = []
  for i, cD in enumerate(cDs):
      if i >= num_levels - 3:
          scale = (i - (num_levels - 3) + 1) / 3
          thresh = base_thresh * scale
          new_cD = pywt.threshold(cD, thresh, mode='soft')
      else:
          new_cD = cD
      new_cDs.append(new_cD)
  new_coeffs = [cA] + new_cDs
  denoised = pywt.waverec(new_coeffs, wavelet)

  if denoised is not None:
    return denoised[:len(signal)]
  else:
    return signal


def preprocess_signal(signal, fs=360):

  filtered = cheby2_bandpass_filter(signal, fs)
  denoised = wavelet_denoise(filtered)

  return denoised