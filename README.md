# ENERGY PREDICTOR OF PROTEIN SEQUENCES

## Synopsis
Program that takes as input a fasta file, and predicts the energy content of the proteins contained in the fasta file. The energy content, based on the PROFASI force fields, is: Hydrophobicity, Sidechain charge, and Helix, Sheet and Coil hydrogen bond. There is a vector of 5 values for each amino-acid. 

Program done during the Master's thesis: " Predicting the energy content of proteins using deep learning"

Transforms FASTA sequences to sequences of matrices of 20X1. These are introduced in a bidirectional GRU neural network complemented with two previous feed-forward layers and three posterior feed-forward layers.

## Author 
Alfred Ferrer Florensa, KU Bioinformatics

