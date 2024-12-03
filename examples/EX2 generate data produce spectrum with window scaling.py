import FourierTransformHelperLib as ft
import numpy as np
import matplotlib.pyplot as plt

"""
FourierTransformHelperLib - Example 2
Generate a signal and produce a spectrum magnitude output
also includes adding window and window correction scaling factors
"""

# Generate a 2000 Hz Test Signal Tone with a 1.5 Vrms Amplitude
time, sig = ft.tone_sampling_points(amplitude=1.5, frequency=2000, sampling_frequency=10000, points=1000, phase=0)

# Plot input signal
plt.figure(1)
plt.plot(time, sig)
plt.title('Input Signal (Zoom in for a closer look)')
plt.xlabel('Time [Seconds]')
plt.ylabel('Amplitude [Volts]')

# Generate the Window Coefficients for a Hamming window
wc = ft.window(window_type='hamming', points=1000)

# Calculate the window scale factor
wc_sf = ft.window_scale_signal(wc)

# Apply window to signal
sig_w = sig * wc

# Plot windowed data prior to Transform
plt.figure(2)
plt.plot(time, sig_w)
plt.title('Windowed Input Signal (Zoom in for a closer look)')
plt.xlabel('Time [Seconds]')
plt.ylabel('Amplitude [Volts]')

# Preform Fourier Transform and convert amplitude to magnitude
ft_sig = ft.ft_cpx(sig_w)
y = ft.complex_to_mag(ft_sig)

# Correct for the spectrum amplitude change because of the applied window
# by applying the window scale factor
y = y * wc_sf

# Get X Scale
x = ft.frequency_bins(sampling_rate=10000, signal_length=len(sig))

# Plot Spectrum
plt.figure(3)
plt.semilogy(x, y)
plt.title('Scaled Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [Vrms]')
plt.show()
