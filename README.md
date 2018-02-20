# ENERGY PREDICTOR OF PROTEIN SEQUENCES

## Synopsis
Program that takes as input a fasta file, and predicts the energy content of the proteins contained in the fasta file. The energy content, based on the PROFASI force fields, is: Hydrophobicity, Sidechain charge, and Helix, Sheet and Coil hydrogen bond. There is a vector of 5 values for each amino-acid. 

Program done during the Master's thesis: " Predicting the energy content of proteins using deep learning"

Transforms FASTA sequences to sequences of matrices of 20X1. These are introduced in a bidirectional GRU neural network complemented with two previous feed-forward layers and three posterior feed-forward layers.

## Files
Files contained in the "energy_predictor" repository:

### aminoacid_dictionary.pkl  
	-> Dictionary necessary for transform FASTA sequences to numpy arrays
### neural_netGRU.py    
	-> File containing the structure of the neural network called by prediction-energies.py
### prediction-energies.py 
	-> Main program described above
### utils.py
	-> File that contains necessary functions
### dropout_seq.py
	-> File that contains the code for the sequential dropout
### predicted_proteins 
	-> Folder that will contain the output of the prediction-energies.py. It also contains a folder with the strange
	proteins that have not been able to process.
### try.fasta   
	-> Fasta file to try the program 
### weights_neuralnetwork.pkl
	-> Weights and preprocessing/postprocessing values for the neural network

## Author 
Alfred Ferrer Florensa, KU Bioinformatics

## Contributors
Thomas Hamelryck, KU Bioinformatics
Jesper Foldager, KU Bioinformatics
Jose Juan Almagro Armenteros, DTU Bioinformatics
