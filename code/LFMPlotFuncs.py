import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def spectrum_2channels(arr, fs):
    for chan in [0, 1]:
        plt.figure(chan)
        plt.specgram(x=arr[:,chan], Fs=fs)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        chan+=1
        plt.show(block=False)


def plot_xcorr(tx_lfm, rx_lfm):
    plt.figure(0)
    plt.xcorr(
        x=rx_lfm[:,0]+1j*rx_lfm[:,1],
        y=tx_lfm[:,0]+1j*tx_lfm[:,1]
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.show(block=False)
