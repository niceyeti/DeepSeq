"""
Using the torch-based GRU, implements a word prediction model using vector word embeddings as input.

Using an algorithm such as word2vec, each word in a training corpus is converted to a k-dimensional vector.
These inputs are fed to a recurrent network, whose outputs are one-hot term predictions.

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

def usage():
	print("Usage: python3 word_prediction.py")
	print("Params (these apply differently to selected models): -eta,\
						-maxEpochs/-epochs, \
						-hiddenUnits,\
						-bpStepLimit,\
						-numSequences,\
						-miniBatchSize\
						-maxSeqLen,\
						-clip,\
						-embedModel=[path to embedding model],\
						-trainFile=[path to training text]")
	print("Models: --torch-gru=[rnn or gru], --numpy-rnn, --custom-torch-rnn")
	print("Suggested example params: python3 word_prediction.py  -maxEpochs=100000 -momentum=0.9 -eta=1E-3 --torch-gru -batchSize=10 -numHiddenLayers=2")

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
	clip = -1.0 #Only applies to pytorch rnn/gru's, to mitigate exploding gradients, but I don't suggest using it. 
	saveMinWeights = "--saveMinWeights" in sys.argv
	for arg in sys.argv:
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

	trainPath = "../data/treasureIsland_normalized.txt"
	modelPath = "../data/treasure_island_wordtovec_100iter_150d_10w_5min_cbow"

	ignoreIndex = -1

	print("Building vector dataset...")
	dataset, vecModel = GetEmbeddedDataset(trainPath, modelPath, limit=numSequences, maxSeqLen=20, ignoreIndex=ignoreIndex)
	print("Converting to tensor batch data...")
	batchedData = batchifyTensorData(dataset, batchSize=miniBatchSize)
	#print("Randomizing dataset...")
	#random.shuffle(dataset)

	print(dataset[100])

	useRNN = False
	#print(str(dataset[0]))
	#print("SHAPE: num examples={} xdim={} ydim={}".format(len(dataset), dataset[0][0][0].shape, dataset[0][0][1].shape))
	xDim = vecModel.layer1_size
	yDim = len(vecModel.wv.vocab)

	#Try these params: python3 BPTT.py  -maxEpochs=100000 -momentum=0.9 -eta=1E-3 --torch-gru -batchSize=10 -numHiddenLayers=2
	gru = EmbeddedGRU(xDim, hiddenUnits, yDim, numHiddenLayers=numHiddenLayers, batchFirst=True, clip=clip, useRNN=useRNN)
	print("Training...")
	gru.train(batchedData, epochs=maxEpochs, batchSize=miniBatchSize, torchEta=eta)
	gru.generate(reverseEncoding,30,30,stochastic=True)
	gru.generate(reverseEncoding,30,30,stochastic=False)

if __name__ == "__main__":
	main()

