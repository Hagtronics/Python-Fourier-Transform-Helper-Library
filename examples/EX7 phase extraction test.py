import FourierTransformHelperLib as ft
import numpy as np
import matplotlib.pyplot as plt

"""
FourierTransformHelperLib - Example 7
Generate a signal with a phase shift, recover the signals phase
"""

# Generate a 2000 Hz Test Signal Tone with a 1.5 VRMS Amplitude, phase 25 degrees
time, sig_phase = ft.tone_sampling_points(amplitude=1.5,
                                          frequency=2000,
                                          sampling_frequency=10000,
                                          points=1000,
                                          phase=25)

# Generate a 1000 Hz reference signal, phase = 0 degrees
time, sig_ref = ft.tone_sampling_points(amplitude=1.5,
                                        frequency=1000,
                                        sampling_frequency=10000,
                                        points=1000, phase=0)

# Add the signals together
sig_combined = sig_ref + sig_phase

# Preform Fourier Transform and convert amplitude to magnitude
ft_sig_combined = ft.ft_cpx(sig_combined)

# Convert complex spectrum output to phase degree values
ph = ft.complex_to_phase_degrees(ft_sig_combined)

# 1000 Hz is at bin number 100, get phase at this bin for reference
ph_ref = ph[100]

# 2000 Hz is at bin number 200, get phase at this bin
ph_sig = ph[200]

# Get phase shift delta of 1000 Hz Signal to 2000Hz signal
# This will be equal to 25 Degrees since that is what we programmed the difference to be above
ph_shift = ph_sig - ph_ref

print(f'Phase Shift = {ph_shift} Degrees\n')
input('Press ENTER key to continue...')
