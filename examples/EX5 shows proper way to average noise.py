import FourierTransformHelperLib as ft
import numpy as np
import matplotlib.pyplot as plt


####### TODO - Finish this


"""
FourierTransformHelperLib - Example 5
Generate noise signal, and produce a averaged spectrum magnitude output
Shows proper way to scale and average noise
"""

# Noise generated will be 5 nVrms / rt-Hz
# Sampling Frequency will be 100,000 Samples per second
# FFT Points will be 2048
# We will use a 'hann' or 'hanning' window for no paticular reason
# Average 1000 FT spectrums to reduce the noise deviation and get the true 'average noise'


# Generate the Window Coefficients for a Hann window
wc = ft.window(window_type='hann', points=2048)

# Calculate the window scale factor for NOISE!
wc_sf = ft.window_scale_noise(wc, sampling_frequency=100000)

# Loop 1000 times averaging the generated random noise
# The proper way to average noise is to use the Magnitude Squared Format (Mag2)
sum_spectrum2 = []
averages = 0

for i in range(1000):
    sig = ft.noise_psd(amplitude_psd=5.0e-9, sampling_frequency=100000, points=2048)

    # Apply window to input signal
    sig_w = sig * wc

    # Preform Fourier Transform and convert amplitude to magnitude squared
    ft_cpx = ft.ft_cpx(sig_w)
    ft_mag2 = ft.complex_to_mag2(ft_cpx)

    if i == 0:
        sum_spectrum2 = ft_mag2.copy()
    else:
        sum_spectrum2 = ((sum_spectrum2 + ft_mag2)).copy()

    averages += 1


# Calculate the real average of the Magnitude Squared Spectrum (mag2)
averaged_spectrum2 = sum_spectrum2 / averages

# Now convert the averaged spectrum Mmeraged_spectrum2)
averaged_spectrum = ft.mag2_to_mag(averaged_spectrum2)

# Apply the proper NOISE window scale factor to the result
# so that the amplitude is correct
averaged_spectrum = averaged_spectrum * wc_sf

# Since the noise generated is flat or 'white' we can just use average the resulting
# spectrum. Be sure to slice off the first and last 10 bins to remove the DC and Fs/2
# bin effects.

# As we increase the number of averages above, this will get closer and closer to 5 nV/rt-Hz
average_noise_result = np.average(averaged_spectrum[10:-10])

print(f'Average Noise Value = {average_noise_result}  Volts / rt-Hz')
input('Press ENTER key to continue...')
