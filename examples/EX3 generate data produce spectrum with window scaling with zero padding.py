import FourierTransformHelperLib as ft
import numpy as np
import matplotlib.pyplot as plt

"""
FourierTransformHelperLib - Example 3
Generate a signal and produce a spectrum magnitude output
also includes adding window and window correction scaling factors
also includes zero padding to the Fourier Transform
"""

# Generate a 2000 Hz Test Signal Tone with a 1.0 VRMS Amplitude
time, sig = ft.tone_sampling_points(amplitude=1.0, frequency=2000, sampling_frequency=10000, points=1000, phase=0)

# Plot input signal
plt.figure(1)
plt.plot(time, sig)
plt.title('Input Signal (Zoom in for a closer look)')
plt.xlabel('Time [Seconds]')
plt.ylabel('Amplitude [Volts]')


# Generate the Window Coefficients for a HP Flattop window
wc = ft.window(window_type='fthp', points=1000)

# Calculate the window scale factor
wc_sf = ft.window_scale_signal(wc)

# Apply window to input signal
sig_w = sig * wc

# Preform Fourier Transform and convert amplitude to magnitude
# add a 1000 points of zero padding, so total FT length is: 1000 + 3096 = 4096
# Note: The scaling change due to zero padding is automatically accounted for by
# the function ft_raw()
ft_sig = ft.ft_cpx(sig_w, zero_padding=3096)
y = ft.complex_to_mag(ft_sig)

# Correct for the spectrum amplitude change because of the applied window
# by applying the window scale factor
y = y * wc_sf

# Get X Scale
# Note: Need to add the zero padding length here also
x = ft.frequency_bins(sampling_rate=10000, signal_length=len(sig), zero_padding_length=3096)

# Plot Spectrum
plt.figure(2)
plt.semilogy(x, y)
plt.ylim(1e-5, 10.0)
plt.title('Scaled Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [Vrms]')

plt.show()
