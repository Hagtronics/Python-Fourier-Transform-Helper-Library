import FourierTransformHelperLib as ft
import numpy as np
import matplotlib.pyplot as plt

"""
FourierTransformHelperLib - Example 6
Generate a signal with noise and produce a spectrum magnitude output
Simulates a real signal
"""

# Generate a 2000 Hz Test Signal Tone with a 1.0 VRMS Amplitude
time, sig = ft.tone_sampling_points(amplitude=1.0, frequency=2000, sampling_frequency=10000, points=1000, phase=0)

# Generate random noise 10 times less than the signal
noise = ft.noise_rms(amplitude_rms=1.0/10.0, points=1000)

# Add signal and noise
sig_new = sig + noise

# Plot input signal
plt.figure(1)
plt.plot(time, sig_new)
plt.title('Input Signal (Zoom in for a closer look)')
plt.xlabel('Time [Seconds]')
plt.ylabel('Amplitude [Volts]')

# Generate the Window Coefficients for a HP Flattop window
wc = ft.window(window_type='fthp', points=1000)

# Calculate the window scale factor
wc_sf = ft.window_scale_signal(wc)

# Apply window to input signal
sig_w = sig_new * wc

# Preform Fourier Transform and convert amplitude to magnitude
# Note: The scaling change due to zero padding is automatically accounted for by
# the function ft_raw()
ft_sig = ft.ft_cpx(sig_w)
y = ft.complex_to_mag(ft_sig)

# Correct for the spectrum amplitude change because of the applied window
# by applying the window scale factor
y = y * wc_sf

# Now convert the magnitude signal to dBV units (Log Scale)
y_dBV = ft.mag_to_dBV(y)

# Get X Scale
# Note: Need to add the zero padding length here also
x = ft.frequency_bins(sampling_rate=10000, signal_length=len(sig))

# Plot Spectrum
plt.figure(2)
plt.plot(x, y_dBV)

plt.title('Scaled Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dBV]')
plt.show()
