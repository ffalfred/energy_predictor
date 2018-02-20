# ENERGY PREDICTOR OF PROTEIN SEQUENCES

## Synopsis
Program that takes as input a fasta file, and predicts the energy content of the proteins contained in the fasta file. The energy content, based on the PROFASI force fields, is: Hydrophobicity, Sidechain charge, and Helix, Sheet and Coil hydrogen bond. There is a vector of 5 values for each amino-acid. 

Program done during the Master's thesis: " Predicting the energy content of proteins using deep learning"

Transforms FASTA sequences to sequences of matrices of 20X1. These are introduced in a bidirectional GRU neural network complemented with two previous feed-forward layers and three posterior feed-forward layers.

## Options of use
Options of prediction-energies.py:

### -in: 
fasta file containing the protein sequences. Example: try.fasta

### -p: 
dictionary file containing the preprocessing values and weights for the neural network. It has to be "weights_neuralnetwork"

### -o: 
name of the output file. It will be placed in predicted_proteins. The output file is in format '.npz'. It contains the arrays:
	
	->"id": id of the protein
	->"aminoacid": sequence of amino-acids of the proteins
	->"x": sequence of amino-acid in the binary array form (nº of sequences x max length of protein x 20). In mask format.
	->"mask_seq": mask sequence used for the neural network (nº of sequences x max length of protein x 1).  
	->"predicted": vector of the energies (nº of sequences x max length of protein x 5). In mask format.

### -s: 
name of the output file with proteins that has not been processed due to the length or because it has certain amino-acids

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
	
## Packages and recommendations

Packages required (version):

	-> numpy: (1.13.1)

	-> theano: (0.9.0)
	
	-> lasagne (0.2.dev1)
	
	-> time
	
	-> _pickle
	
	-> Bio (1.70)
	
	-> argparse (1.1)
	
	-> random
	
	-> os
	
	-> math
	
	-> importlib

Notice that, as it uses THEANO, the /.theanorc file must contain:

	[lib]
	cnmem=.75

	[global]
	floatX = float32
	device = cuda

Or add before "python3":

	THEANO_FLAGS='floatX=float32,device=cuda,lib.cnmem=.75'

## Author 
Alfred Ferrer Florensa, KU Bioinformatics

## Contributors
Thomas Hamelryck, KU Bioinformatics
Jesper Foldager, KU Bioinformatics
Jose Juan Almagro Armenteros, DTU Bioinformatics
