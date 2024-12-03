import FourierTransformHelperLib as ft
import numpy as np
import matplotlib.pyplot as plt
import math

"""
FourierTransformHelperLib - Example 8
Generate a IQ Data (Complex) and produce a spectrum magnitude output
also includes adding window and window correction scaling factors
"""

# Generate a IQ Test Signal
POINTS = 4096
CYCLES = 102
phase_inc = 2.0 * math.pi * CYCLES/POINTS

ph = []
for k in range(POINTS):
    ph.append(phase_inc * k)

# Make I and Q arrays
i = np.cos(ph)
q = np.sin(ph)

# Now combine the I and Q parts to a single complex signal vector
iq_sig = i + (1j * q)


# Plot i and q parts of input signal
plt.figure(1)
plt.plot(i)
plt.plot(q)
plt.title('Input Signal I&Q (Zoom in for a closer look)')
plt.xlabel('Time [Seconds]')
plt.ylabel('Amplitude [Volts]')


# Generate the Window Coefficients for a Blackman-Harris 92 window
wc = ft.window(window_type='bh92', points=POINTS)

# Calculate the window scale factor
wc_sf = ft.window_scale_signal(wc)

# Apply window to the complex IQ signal
iq_sig_w = iq_sig * wc

# Preform Fourier Transform on IQ signal and convert amplitude to magnitude
ft_sig = ft.ft_cpx(iq_sig_w)
y = ft.complex_to_mag(ft_sig)

# Correct for the spectrum amplitude change because of the applied window
# by applying the window scale factor
y = y * wc_sf


# Plot resulting Spectrum
# In this example the peak in be in bin 102 (corresponds with the value: CYCLES above)
plt.figure(2)
plt.semilogy(y)

plt.title('Scaled Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [Vrms]')
plt.show()
