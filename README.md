# Python-Fourier-Transform-Helper-Library  
  
## Overview:  
A Python Library to help make properly scaled Fourier Transforms including utility functions.
The vast majority of code that you will find in Commercial packages, Open Source libraries, Textbooks, and on the Web is
simply unsuited for the task of making a "Properly Scaled" Fourier Transform, and takes hours of further tweaking to get a
classic and properly scaled spectrum plot. You know: Proper “Amplitude” and no more “Negative frequencies”, etc. This
library is a solution that works "Out of the Box" and also includes example code to get you easily started.

## Introduction:
There is a real need in Python 3 for a ready-to-use Fourier Transform Library that users can take right out of the box and
perform Fourier Transforms (FT), and get a classical, properly scaled spectrum versus frequency plot.
The vast majority of code that you will find in Commercial packages, Open Source libraries, Textbooks, and on the Web is
simply unsuited for this task and takes hours of further tweaking to get a classic and properly scaled spectrum plot. You
know: Proper “Amplitude” and no more “Negative frequencies”, etc.  
  
## What “Fourier Transform Helper Lib” Does:  
FourierTransformHelperLib.py (FTHL) has several main parts, but its basic goal is to allow a real Fourier
Transform to be performed on a real or complex time series input array, resulting in a usable classic spectrum output
without any further tweaking required by the user.  
  
Step by step examples are given for many use cases showing how to,  
1) Generate test signals (Which can be replaced by your real signals later).
2) Properly apply a Window to the input data, using any of the 32 included windows.
3) Preform the Fourier Transform.
4) Properly scale the Fourier Transform to correct for the applied Window in step 2.
5) Convert the Fourier Transform to Magnitude or dBV format.
6) Plot a nice frequency amplitude display of the real frequencies (i.e. a Classical Spectrum Plot)
  
## What “Fourier Transform Helper Lib” Does Not Do:  
With all this windowing, scaling, and slicing of the raw Fourier Transform data into a usable positive frequency only display,
it is not possible to go backward and perform a proper Inverse Fourier Transform on the FFT output. This is because of the
way the Fourier Transforms work. The transforms basic math requires that the total energy in a forward and inverse
transform be maintained [2]. After all the windowing, scaling, etc. This would have to be backed out somehow to get back
to the original signal and this is not the purpose of this library at all. As shown above, the purpose of this library is to make
a usable and correctly scaled Forward Transform with the minimum user required steps.
  
## More:  
See the full user guide in the 'docs' directory.  
The source code is in the 'src' directory.  
The examples are in the 'examples' directory.  
