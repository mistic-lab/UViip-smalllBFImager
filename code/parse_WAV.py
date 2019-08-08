import numpy as np
from scipy.io import wavfile


def parse_WAV(tx_file_path='../MATLAB/Transmission-files/100kHz_0.5s_fs200k.wav'):
    fs, data = wavfile.read(tx_file_path)
    return fs, data
