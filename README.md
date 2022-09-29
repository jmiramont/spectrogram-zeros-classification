# Unsupervised classification of the spectrogram zeros

Juan M. Miramont, François Auger, Marcelo A. Colominas, Nils Laurent, Sylvain Meignen

## Abstract

The zeros of the spectrogram have proven to be a relevant feature to describe the time-frequency structure of a signal, originated by the destructive interference between components in the time-frequency plane. In this work, a classification of these zeros in three types is introduced, based on the nature of the components that interfere to produce them. Echoing noise-assisted methods, a classification algorithm is proposed based on the addition of independent noise realizations to build a 2D histogram describing the stability of zeros. Features extracted from this histogram are later used to classify the zeros using a non-supervised clusterization algorithm. A denoising approach based on the classification of the spectrogram zeros is also introduced. Examples of the classification of zeros are given for synthetic and real signals, as well as a performance comparison of the proposed denoising algorithm with another zero-based approach.

## Code

This repository contains the Matlab and Python code used in the paper. Each matlab script is named with the figure that generates. The main functions are:

1. ```compute_zeros_histogram.m```
2. ```classify_spectrogram_zeros```
3. ```classified_zeros_denoising.m```

Part of the detection tests used in the paper were implemented in Python. The folder ```detection_tests``` contains notebooks displaying the results.
