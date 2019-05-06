"""
Using the torch-based GRU, implements a word prediction model using vector word embeddings as input.

Using an algorithm such as word2vec, each word in a training corpus is converted to a k-dimensional vector.
These inputs are fed to a recurrent network, whose outputs model one-hot term predictions.

It would also be possible to predict word embeddings on the output, but the loss function would likely have to be
different. Predicting a k-dimensional real-valued vector seems less stable than clamping to one-hot output loss
to drive the model toward target terms.

This model requires a pre-built vector-embedding model to convert input terms to vector embeddings, such that
every training term must be contained by the model, otherwise it has no embedding description.
I have a Word2Vec.py script in the repo DoubleSecretProbation/RNN for generating such a model, and may move it here
in the future.
"""

import sys
import random
import torch
from numpy_rnn import NumpyRnn
import matplotlib.pyplot as plt
from custom_torch_rnn import *
from embedded_word_prediction_rnn import *
from data_lib import *
from embedded_dataset import EmbeddedDataset

TORCH_DTYPE = torch.float32
torch.set_default_dtype(TORCH_DTYPE)

def usage():
	print("Usage: python3 word_prediction.py")
	print("""Params (these apply differently to selected models):
	-eta
	-maxEpochs/-epochs 
	-hiddenUnits
	-bpStepLimit
	-numSequences
	-miniBatchSize
	-maxSeqLen
	-clip
	-modelPath=[path to embedding model]
	-trainPath=[path to training text]
	--useL2Norm: normalize word vectors in training inputs (currently inefficient, as term vector norms are not memo'ized)
""")
	print("Models: --torch-gru=[rnn or gru], --numpy-rnn, --custom-torch-rnn")
	print("Suggested example params: python3 word_prediction.py  -maxEpochs=100000 -momentum=0.9 -eta=1E-2 -batchSize=3 -numHiddenLayers=1 -hiddenUnits=300")

def main():
	eta = 1E-5
	hiddenUnits = 50
	numHiddenLayers = 1
	maxEpochs = 500
	miniBatchSize = 2
	momentum = 1E-5
	bpStepLimit = 4
	maxSeqLen = 200
	numSequences = 100000
	useRNN = False
	optimizer = "adam" #TODO: Figure out the "generally best" optimizer, or default to sgd.
	clip = -1.0 #Only applies to pytorch rnn/gru's, to mitigate exploding gradients, but I don't suggest using it. 
	#saveMinWeights = "--saveMinWeights" in sys.argv
	useL2Norm = "--useL2Norm" in sys.argv
	for arg in sys.argv:
		if "-trainPath=" in arg:
			trainPath = arg.split("=")[-1]
		if "-modelPath=" in arg:
			modelPath = arg.split("=")[-1]
		if "-hiddenUnits=" in arg:
			hiddenUnits = int(arg.split("=")[-1])
		if "-eta=" in arg:
			eta = float(arg.split("=")[-1])
		if "-momentum=" in arg:
			momentum = float(arg.split("=")[-1])
		if "-bpStepLimit=" in arg:
			bpStepLimit = int(arg.split("=")[-1])
		if "-maxEpochs=" in arg or "-epochs=" in arg:
			maxEpochs = int(arg.split("=")[-1])
		if "-miniBatchSize=" in arg or "-batchSize" in arg:
			miniBatchSize = int(arg.split("=")[-1])
		if "-numSequences" in arg:
			numSequences = int(arg.split("=")[-1])
		if "-numHiddenLayers=" in arg:
			numHiddenLayers = int(arg.split("=")[-1])
		if "-clip=" in arg:
			clip = float(arg.split("=")[-1])
		if "-maxSeqLen" in arg:
			maxSeqLen = int(arg.split("=")[-1])
		if "--rnn" in arg.lower() or "--usernn" in arg.lower():
			useRNN = True
		if "-optimizer=" in arg:
			optimizer = arg.split("=")[-1]
	
	trainPath = "../data/treasureIsland_normalized.txt"
	modelPath = "../data/treasure_island_wordtovec_100iter_150d_10w_5min_cbow"
	#trainPath = "../data/wapo.txt"
	#modelPath = "../../VecSent/models/big_model.d2v"
	ignoreIndex = -1

	dataset = EmbeddedDataset( \
		trainPath, \
		modelPath, \
		batchSize = miniBatchSize, \
		batchCacheSize = 200, \
		torchDtype = TORCH_DTYPE, \
		maxSeqLength = 200, \
		minSeqLength = 5, \
		useL2Norm = useL2Norm)

	xDim = dataset.Model.layer1_size
	yDim = len(dataset.Model.wv.vocab)
	print("X dim: {}  Class size: {}".format(xDim, yDim))

	#Try these params: python3 BPTT.py  -maxEpochs=100000 -momentum=0.9 -eta=1E-3 --torch-gru -batchSize=10 -numHiddenLayers=2
	gru = EmbeddedGRU( \
		xDim, \
		hiddenUnits, \
		yDim, \
		numHiddenLayers=numHiddenLayers, \
		batchFirst=True, \
		clip=clip, \
		useRNN=useRNN)
	print("Training...")

	gru.train(dataset, epochs=maxEpochs, torchEta=eta, momentum=momentum, optimizerStr=optimizer)
	#gru.Save()
	#gru.Read()

	gru.beamGenerate(dataset.Model, k=1, beamWidth=100, numSeqs=10, seqLen=12)
	gru.generate(dataset.Model, 30, 30, stochasticChoice=True)
	gru.generate(dataset.Model, 30, 30, stochasticChoice=False)

if __name__ == "__main__":
	main()

