import numpy as np
import matplotlib.pyplot as plt

# Junk I need
fs = 1.024e6
timestep = 1/fs


# Used Config2/3700KHZ_CW.dat
run datToArrays.py # Pulls out arr00,arr01...arr99 at 3.7MHz

# fft that thing
arrFFT01 = abs(np.fft.fft(arr01))  # abs to take magnitude

n = arr01.size  # 5003
freqs = np.fft.fftfreq(n, d=timestep)
freqshifted = np.fft.fftshift(freqs)

# Sanity check
len(arr01)  # 5003
len(arrFFT01)  # 5003
len(freqs)  # 5003
len(freqshifted)  # 5003


# Plot it and pray
plt.plot(freqshifted, arrFFT01)
plt.show(block=False)
