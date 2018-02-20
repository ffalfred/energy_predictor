-----------------------------------------------------------------------------------------
README FILE FOR THE PROGRAM 'prediction-energies.py', developed by Alfred Ferrer Florensa
-----------------------------------------------------------------------------------------
Program that takes as input a fasta file, and predicts the energy content of the proteins
contained in the fasta file. The energy content is: Hydrophobicity, Sidechain charge, and
Helix, Sheet and Coil hydrogen bond. There is a vector of 5 values for each amino-acid. 

Program done during the Master's thesis: " Predicting the energy content of proteins using
deep learning"

----------------------------------

Example of usage:

$python3 prediction-energies.py -in try.fasta -p weights_neuralnetwork -o try

----------------------------------

Options of prediction-energies.py:
	-in: fasta file containing the protein sequences. Example: try.fasta

	-p: dictionary file containing the preprocessing values and weights for the neural 
            network. It has to be "weights_neuralnetwork"

	-o: name of the output file. It will be placed in predicted_proteins. The output
	    file is in format '.npz'. It contains the arrays:
		*"id": id of the protein
		*"aminoacid": sequence of amino-acids of the proteins
		*"x": sequence of amino-acid in the binary array form (nº of sequences x max length of protein x 20). In mask format.
		*"mask_seq": mask sequence used for the neural network (nº of sequences x max length of protein x 1).  
		*"predicted": vector of the energies (nº of sequences x max length of protein x 5). In mask format.

	-s: name of the output file with proteins that has not been processed due to the
	    length or because it has certain amino-acids

----------------------------------

Files contained in the "energy_predictor" folder:

	aminoacid_dictionary.pkl  
	-> Dictionary necessary
	neural_netGRU.py    
	-> File containing the structure of the neural network called by prediction-energies.py
	prediction-energies.py 
	-> Main program described above
	utils.py
	-> File that contains necessary functions
	dropout_seq.py
	-> File that contain the code for the sequential dropout
        predicted_proteins 
	-> Folder that will contain the output of the prediction-energies.py
	try.fasta   
	-> Fasta file to try the program 
	weights_neuralnetwork.pkl
	-> Weights and preprocessing values for the neural network

---------------------------------

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

         



