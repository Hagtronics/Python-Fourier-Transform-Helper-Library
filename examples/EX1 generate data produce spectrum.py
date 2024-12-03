import FourierTransformHelperLib as ft
import numpy as np
import matplotlib.pyplot as plt

"""
FourierTransformHelperLib - Example 1
Generate a signal and produce a spectrum magnitude output
"""

# Generate a 2000 Hz Test Signal Tone with a 1.5 VRMS Amplitude
time, sig = ft.tone_sampling_points(amplitude=1.5, frequency=2000, sampling_frequency=10000, points=1000, phase=0)

# Plot input signal
plt.figure(1)
plt.plot(time, sig)
plt.title('Input Signal (Zoom in for a closer look)')
plt.xlabel('Time [Seconds]')
plt.ylabel('Amplitude [Volts]')

# Preform Fourier Transform and convert amplitude to magnitude
ft_sig = ft.ft_cpx(sig)
y = ft.complex_to_mag(ft_sig)

# Get X Scale
x = ft.frequency_bins(sampling_rate=10000, signal_length=len(sig))

# Plot Spectrum
plt.figure(2)
plt.semilogy(x, y)

plt.title('Scaled Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [Vrms]')
plt.show()
