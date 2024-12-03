""" ---------------------------------------------------------------------------

Author: Steve Hageman 2022

Description: A Fourier Transform Library for producing proper scaled
Fourier Transforms for signal or noise analysis, with proper scaling functions
for handling any windows.

Dependencies / tested with: 
    Python      3.9.9
    scipy.fft   1.8.0
    numpy       1.22.2

License: Total freeware, but remember this was: Written by an infinite number
of Monkeys, in an infinite amount of time, So BEWARE as Monkeys have no idea
how to type. Hence this code is probably full of issues.

Version History:
    15Apr22 - Revision 0.1

-------------------------------------------------------------------------------
"""

import numpy as np
from scipy.fft import fft


# =====[ Signal Generation Functions ]=========================================
def tone_cycles(amplitude, cycles, points, phase):
    """
    Generates a sine wave tone, of a specified number of whole or partial cycles.

    Parameters
    ----------
    amplitude : float
        The desired amplitude in RMS volts.
    cycles : float
        The number of whole or partial sinewave cycles to generate.
    points : int
        Number of points total to generate.
    phase : float
        Phase angle in degrees to generate. The default is 0.0.

    Returns
    -------
    numpy array of the generated tone values.

    """
    # Convert phase to radians
    ph_r = phase * np.pi / 180.0

    amp_pk = np.sqrt(2.0) * amplitude
    t = np.linspace(0, 1, points)
    y = amp_pk * np.sin((2.0 * np.pi * t) * cycles + ph_r)

    return y


def tone_sampling_points(amplitude, frequency, sampling_frequency, points, phase):
    """
    Generates a sine wave tone, like it would be if sampled at a specified
    sampling frequency.

    Parameters
    ----------
    amplitude : float
        The desired amplitude in RMS volts.
    frequency : float
        Frequency in Hz of the generated tone.
    sampling_frequency : float
        Sampling frequency in Hz.
    points : int
        Number of points to generate.
    phase : float
        Phase angle in degrees to generate.

    Returns
    -------
    numpy tuple array of the time scale (x) and generated tone values (y).

    """
    # Convert phase to radians
    ph_r = phase * np.pi / 180.0

    amp_pk = np.sqrt(2.0) * amplitude
    t = np.linspace(0, points-1, points) / sampling_frequency
    y = amp_pk * np.sin(2.0 * np.pi * frequency * t + ph_r)

    return t, y


def tone_sampling_duration(amplitude, frequency, sampling_frequency, duration, phase):
    """
    Generates a sine wave tone, like it would be if sampled at a specified
    duration.

    Parameters
    ----------
    amplitude : float
        The desired amplitude in RMS volts.
    frequency : float
        Frequency in Hz of the generated tone.
    sampling_frequency : float
        Sampling frequency in Hz.
    duration : float
        Number of seconds to generate.
    phase : float
        Phase angle in degrees to generate.

    Returns
    -------
    numpy tuple array of the time scale (x) and generated tone values (y).

    """
    # Calculate points from duration
    n = sampling_frequency * duration

    t, y = tone_sampling_points(amplitude, frequency, sampling_frequency, n, phase)

    return t, y


def noise_psd(amplitude_psd, sampling_frequency, points):
    """
    Generates a random noise stream of a given power spectral density
    in Vrms/rt-Hz. The noise is normally distributed, with a mean of 0.0 Volts.

    Parameters
    ----------
    density : float
        The desired noise density of the generated signal Vrms/rt-Hz.
    sampling_frequency : float
        Sampling frequency in Hz.
    number : int
        Number of points total to generate.

    Returns
    -------
    numpy array of the generated noise.

    """
    a_rms = amplitude_psd * np.sqrt(sampling_frequency / 2.0)
    return noise_rms(a_rms, points)


def noise_rms(amplitude_rms, points):
    """
    Generates a random noise stream of a given RMS voltage.
    The noise is normally distributed, with a mean of 0.0 Volts.

    Parameters
    ----------
    amplitude : float
        The desired amplitude in Volts RMS units..
    number : int
        Number of points total to generate.

    Returns
    -------
    numpy array of the generated noise.

    """
    # loc is the mean or DC value of the generated noise
    return np.random.normal(loc=0.0, scale=amplitude_rms, size=points)


# =====[ Window Functions ]====================================================
def window(window_type, points):
    """
    Generates the window coefficients for the specified window type.

    Parameters
    ----------
    window_type : string
        One of the following types of window may be specified,
    points : int
        Number of points total to generate.

    Returns
    -------
    numpy array of the generated window coefficients.
    Returns empty list if error.

    """
    # Windows formulas derived from,
    #   G. Heinzel, A. Rudiger and R. Schilling,
    #       “Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
    #       including a comprehensive list of window functions and some new ﬂat-top windows.”,
    #   Max-Planck-Institut fur Gravitationsphysik, February 15, 2002

    N = points
    if N == 1:
        return [1]

    # Make array of the proper length for cos() functions
    z = 2.0 * np.pi * np.linspace(0, N-1, N) / N

    if window_type == "none" or window_type == "rectangular":
        wc = np.ones(N)

    elif window_type == "bartlett":
        n = np.linspace(0, N-1, N)
        wc = 2/N*(N/2-abs(n-(N-1)/2))

    elif window_type == "welch":
        # n = (0:N-1)';
        n = np.linspace(0, N-1, N)
        wc = 1 - (((2*n)/N) - 1)**2

    elif window_type == "hann" or window_type == "hanning":
        wc = (0.5 - 0.5*np.cos(z))

    elif window_type == "hamming":
        wc = 0.54 - 0.46*np.cos(z)

    elif window_type == "bh92":  # Also known as: Blackman-Harris
        wc = (0.35875 - 0.48829*np.cos(z) + 0.14128*np.cos(2*z) - 0.01168*np.cos(3*z))

    elif window_type == "blackman-nutall":
        wc = (0.3635819 - 0.4891775*np.cos(z) +
              0.1365995*np.cos(2*z) - 0.0106411*np.cos(3*z))

    elif window_type == "nutall3":
        c0 = 0.375
        c1 = -0.5
        c2 = 0.125
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z)

    elif window_type == "nutall3a":
        c0 = 0.40897
        c1 = -0.5
        c2 = 0.09103
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z)

    elif window_type == "nutall3b":
        c0 = 0.4243801
        c1 = -0.4973406
        c2 = 0.0782793
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z)

    elif window_type == "nutall4":
        c0 = 0.3125
        c1 = -0.46875
        c2 = 0.1875
        c3 = -0.03125
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z)

    elif window_type == "nutall4a":
        c0 = 0.338946
        c1 = -0.481973
        c2 = 0.161054
        c3 = -0.018027
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z)

    elif window_type == "nutall4b":
        c0 = 0.355768
        c1 = -0.487396
        c2 = 0.144232
        c3 = -0.012604
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z)

    elif window_type == "nutall4c":
        c0 = 0.3635819
        c1 = -0.4891775
        c2 = 0.1365995
        c3 = -0.0106411
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z)

    elif window_type == "sft3f":
        c0 = 0.26526
        c1 = -0.5
        c2 = 0.23474
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z)

    elif window_type == "sft4f":
        c0 = 0.21706
        c1 = -0.42103
        c2 = 0.28294
        c3 = -0.07897
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z)

    elif window_type == "sft5f":
        c0 = 0.1881
        c1 = -0.36923
        c2 = 0.28702
        c3 = -0.13077
        c4 = 0.02488
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z) + c4*np.cos(4*z)

    elif window_type == "sft3m":
        c0 = 0.28235
        c1 = -0.52105
        c2 = 0.19659
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z)

    elif window_type == "sft4m":
        c0 = 0.241906
        c1 = -0.460841
        c2 = 0.255381
        c3 = -0.041872
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z)

    elif window_type == "sft5m":
        c0 = 0.209671
        c1 = -0.407331
        c2 = 0.281225
        c3 = -0.092669
        c4 = 0.0091036
        wc = c0 + c1*np.cos(z) + c2*np.cos(2*z) + c3*np.cos(3*z) + c4*np.cos(4*z)

    elif window_type == "ftni":
        wc = (0.2810639 - 0.5208972*np.cos(z) + 0.1980399*np.cos(2*z))

    elif window_type == "fthp":
        wc = 1.0 - 1.912510941*np.cos(z) + 1.079173272 * np.cos(2*z) - 0.1832630879*np.cos(3*z)

    elif window_type == "hft70":
        wc = 1 - 1.90796*np.cos(z) + 1.07349*np.cos(2*z) - 0.18199*np.cos(3*z)

    elif window_type == "ftsrs":
        wc = 1.0 - 1.93*np.cos(z) + 1.29*np.cos(2*z) - 0.388 * np.cos(3*z) + 0.028*np.cos(4*z)

    elif window_type == "hft90d":
        wc = 1 - 1.942604*np.cos(z) + 1.340318*np.cos(2*z) - 0.440811*np.cos(3*z) + 0.043097*np.cos(4*z)

    elif window_type == "hft95":
        wc = 1 - 1.9383379*np.cos(z) + 1.3045202*np.cos(2*z) - 0.4028270*np.cos(3*z) + 0.0350665*np.cos(4*z)

    elif window_type == "hft116d":
        wc = 1 - 1.9575375*np.cos(z) + 1.4780705*np.cos(2*z) - 0.6367431 * np.cos(3*z) + 0.1228389*np.cos(4*z) - 0.0066288*np.cos(5*z)

    elif window_type == "hft144d":
        wc = 1 - 1.96760033*np.cos(z) + 1.57983607*np.cos(2*z) - 0.81123644*np.cos(3*z) + 0.22583558*np.cos(4*z) - 0.02773848*np.cos(5*z) + 0.00090360*np.cos(6*z)

    elif window_type == "hft169d":
        wc = 1 - 1.97441842*np.cos(z) + 1.65409888*np.cos(2*z) - 0.95788186*np.cos(3*z) + 0.33673420*np.cos(4*z) - 0.06364621*np.cos(5*z) + 0.00521942*np.cos(6*z) - 0.00010599*np.cos(7*z)

    elif window_type == "hft196d":
        wc = 1 - 1.979280420*np.cos(z) + 1.710288951*np.cos(2*z) - 1.081629853*np.cos(3*z) + 0.448734314*np.cos(4*z) - 0.112376628*np.cos(5*z) + 0.015122992*np.cos(6*z) - 0.000871252*np.cos(7*z) + 0.000011896*np.cos(8*z)

    elif window_type == "hft223d":
        wc = 1 - 1.98298997309*np.cos(z) + 1.75556083063*np.cos(2*z) - 1.19037717712*np.cos(3*z) + 0.56155440797*np.cos(4*z) - 0.17296769663*np.cos(5*z) + 0.03233247087*np.cos(6*z) - 0.00324954578*np.cos(7*z) + 0.00013801040*np.cos(8*z) - 0.00000132725*np.cos(9*z)

    elif window_type == "hft248d":
        wc = 1 - 1.985844164102*np.cos(z) + 1.791176438506*np.cos(2*z) - 1.282075284005*np.cos(3*z) + 0.667777530266*np.cos(4*z) - 0.240160796576*np.cos(5*z) + 0.056656381764*np.cos(6*z) - 0.008134974479*np.cos(7*z) + 0.000624544650*np.cos(8*z) - 0.000019808998*np.cos(9*z) + 0.000000132974*np.cos(10*z)

    else:
        # Unknown window type
        raise ValueError('Unknown window type: ' + window_type)

    return wc


def window_scale_signal(window_coefficients):
    """
    Calculate Signal scale factor from window coefficient array.
    Designed to be applied to the "Magnitude" result.

    Args:
        window_coefficients (float array): window coefficients array

    Returns:
        float: Window scale correction factor for 'signal'
    """
    s1 = sum(window_coefficients)
    s2 = s1 / len(window_coefficients)
    return 1.0/s2


def window_scale_noise(window_coefficients, sampling_frequency):
    """
    Calculate Noise scale factor from window coefficient array.
    Takes into account the bin width in Hz for the final result also.
    Designed to be applied to the "Magnitude" result.

    Args:
        window_coefficients (float array): window coefficients array
        sampling_frequency (_type_): sampling frequency in Hz

    Returns:
        float: Window scale correction factor for 'signal'
    """
    n = len(window_coefficients)
    fbin = sampling_frequency/n

    # Sum of window coefficients squared
    s2 = sum(map(lambda i: i * i, window_coefficients))

    # Correct for bin width
    sf = np.sqrt(1.0 / ((s2 / n) * fbin))

    return sf


def noise_bandwidth(window_coefficients):
    """
    Calculate Normalized, Equivalent Noise BandWidth from window coefficient array.

    Args:
        window_coefficients (float array): window coefficients array

    Returns:
        float: Equivalent Normalized Noise Bandwidth
    """
    s1 = sum(window_coefficients)

    # Sum of window coefficients squared
    s2 = sum(map(lambda i: i * i, window_coefficients))

    s1 = s1 / len(window_coefficients)
    nebw = (s2 / (s1 * s1)) / len(window_coefficients)

    return nebw


# =====[ Fourier Transform Function ]==========================================

def ft_cpx(signal, zero_padding=0):
    """
    Performs Fourier Transform on the input signal.
    Input may be real or complex.
    Input may be any length (not just powers of two).
    Uses SciPy.FFT as the base FFT implementation.

    Args:
        signal (float or Complex array): Input Signal to transform
        zero_padding (int, optional): Additional zeros to pad the signal with. Defaults to 0.

    Returns:
        complex: Resulting Fourier Transform as complex numbers.
        sliced to real frequencies only.
    """
    # Zero pad signal as requested
    z = np.zeros(zero_padding)
    sig = np.append(signal, z)

    # FFT the signal + zero_padding
    fft_result = fft(sig)

    # Slice the real frequencies of the FFT from the result.
    # Convention is to consider Fs/2 to be a negative frequency
    # and hence it is not included in the results.
    # The exact FFT length depends if the input length and if it is even or odd.
    # The len() at this point also includes any zero padding.
    n = len(fft_result)
    if n % 2:
        # is_odd
        fft_len = ((n + 1) // 2) - 1
    else:
        # is_even
        fft_len = (n // 2) - 1

    fft_real_freq = fft_result[0:fft_len]

    # 1 - Multiply all elements by  1/len(signal + zeros)
    sf = np.repeat(1.0 / n, fft_len)
    fft_real_freq = fft_real_freq * sf

    # 2 - Multiply all elements by: (len(signal + zeros))/Len(signal)
    sf = np.repeat(n/len(signal), fft_len)
    fft_real_freq = fft_real_freq * sf

    # 3 - Multiply all but 1st element by: 2/sqrt(2)
    sf = np.repeat(2.0/np.sqrt(2.0), fft_len)
    sf[0] = 1
    fft_real_freq = fft_real_freq * sf

    return fft_real_freq


# =====[ Frequency Span Helper Function ]===============================================

def frequency_bins(sampling_rate, signal_length, zero_padding_length=0):
    """
    Calculates the frequency of each Fourier Transform bin.
    Useful for plotting x axis of a Fourier Transform.

    Args:
        sampling_rate (float): Sampling Frequency in Hz
        signal_length (int): Length of input signal
        zero_padding_length (int, optional): Zero padding length. Defaults to 0.

    Returns:
        float array: Array of frequencies for each of the Fourier Transform Bins.
    """
    # Convention is to consider Fs/2 to be a negative frequency
    # and hence it is not included in the results.
    # The exact FFT length depends if the input length and if it is even or odd.
    # The len() at this point includes any zero padding.
    n = signal_length + zero_padding_length

    if n % 2:
        # is_odd
        fft_len = ((n + 1) // 2) - 1
    else:
        # is_even
        fft_len = (n // 2) - 1

    # Calculate Bin width
    bin_w = sampling_rate / n

    # Make frequency array
    x = np.linspace(0, sampling_rate - bin_w, num=n)

    # Slice array to be the proper length to match FFT length
    x = x[0:fft_len]

    return x


# =====[ Unit Conversion Functions ]====================================================

def complex_to_mag(cpx_arry):
    """Convert Complex Array to Magnitude Format Array.

    Args:
        cpx_arry (complex array): Complex Input Array.

    Returns:
        float array: Magnitude Format Output.
    """
    return np.abs(cpx_arry)


def complex_to_mag2(cpx_arry):
    """Convert Complex Array to Magnitude Squared Format Array.

    Args:
        cpx_arry (complex array): Complex Input Array.

    Returns:
        float array: Magnitude Squared Format Output.
    """
    return complex_to_mag(cpx_arry) ** 2


def complex_to_dBV(cpx_arry):
    """Convert Complex Array to dBV Format Array.

    Args:
        cpx_arry (complex array): Complex Input Array.

    Returns:
        float array: dBV Format Output.
    """
    mag_arry = complex_to_mag(cpx_arry)
    return mag_to_dBV(mag_arry)


def mag_to_mag2(mag_arry):
    """Convert Float Array to Magnitude Squared Format Array.

    Args:
        mag_arry (float array): Float Input Array.

    Returns:
        float array: Magnitude Squared Format Output.
    """
    return np.square(mag_arry)


def mag_to_dBV(mag_arry):
    """Convert Float Array to dBV Format Array.

    Args:
        mag_arry (float array): Magnitude Input Array.

    Returns:
        float array: dBV Format Output.
    """
    # This code makes sure that the log10() is never taken of a
    # negative or zero value in the input array
    minimum_num = np.nextafter(0, 1)
    maximum_num = np.max(mag_arry)
    arry_clipped = np.clip(mag_arry, minimum_num, maximum_num)
    return 20 * np.log10(arry_clipped)


def mag2_to_mag(mag2_arry):
    """Convert Magnitude Squared Float Array to Magnitude Format Array.

    Args:
        mag2_arry (float array): Magnitude Squared Input Array.

    Returns:
        float array: Magnitude Format Output.
    """
    # This code makes sure that the sqrt() is never taken of a
    # negative value in the input array
    minimum_num = 0.0
    maximum_num = np.max(mag2_arry)
    arry_clipped = np.clip(mag2_arry, minimum_num, maximum_num)
    return np.sqrt(arry_clipped)


def mag2_to_dBV(mag2_arry):
    """Convert Magnitude Squared Float Array to dBV Format Array.

    Args:
        mag2_arry (float array): Magnitude Squared Input Array.

    Returns:
        float array: dBV Format Output.
    """
    mag_arry = mag2_to_mag(mag2_arry)
    return mag_to_dBV(mag_arry)


def complex_to_phase_degrees(cpx_arry):
    """Convert Complex Number Input To Equivalent Phase Array In Degrees

    Args:
        cpx_arry (complex array): Input Array Of Complex Numbers.

    Returns:
        float array: Resulting Phase Angle In Degrees of Each Input Point
    """
    return np.angle(cpx_arry, deg=True)


def complex_to_phase_radians(cpx_arry):
    """Convert Complex Number Input To Equivalent Phase Array In Radians

    Args:
        cpx_arry (complex array): Input Array Of Complex Numbers.

    Returns:
        float array: Resulting Phase Angle In Radians of Each Input Point
    """
    return np.angle(cpx_arry, deg=False)


# ----- Fini -----
