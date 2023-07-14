# Unsupervised Classification of the Spectrogram Zeros with an Application to Signal Detection and Denoising

Juan M. Miramont, François Auger, Marcelo A. Colominas, Nils Laurent, Sylvain Meignen

## Abstract

Spectrogram zeros, originated by the destructive interference between the components of a signal in the time-frequency plane, have proven to be a relevant feature to describe the time-varying frequency structure of a signal.
In this work, we first introduce a classification of the spectrogram zeros in three classes that depend on the nature of the components that interfere to produce them.
Then, we describe an algorithm to classify these points in an unsupervised way, based on the analysis of the stability of their location with respect to additive noise. 
Potential uses of the classification of zeros of the spectrogram for signal detection and denoising is finally investigated, and compared with other methods on both synthetic 
and real-world signals.

## Supplementary Material

Extra material can be found in  [```sup_material```](sup_material).

## Code

This repository contains the Matlab and Python code used in the paper. Each Matlab script is named with the figure that generates. The main functions are:

1. ```compute_zeros_histogram.m```
2. ```classify_spectrogram_zeros```
3. ```classified_zeros_denoising.m```

Part of the detection tests used in the paper were implemented in Python. The folder ```detection_tests``` contains notebooks displaying the results.

### Dependencies for Matlab code

To use the functions, you must have the Time-Frequency Toolbox developed by François Auger, Olivier Lemoine,Paulo Gonçalvès and Patrick Flandrin in Matlab's path variable.

You can get a copy of the toolbox from: [http://tftb.nongnu.org/](http://tftb.nongnu.org/).

Figures are printed using the function ```print_figure()```. You can get the newest version from: [https://github.com/rleonarduzzi/matlab-fig-printing](https://github.com/rleonarduzzi/matlab-fig-printing).
