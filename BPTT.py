"""
A variety of raw and library rnn implementations using numpy and pytorch, for fun.
This file is just a main driver of sandbox code, and the params apply unequally to each model.

An implementation of BPTT, because I like the math, but could only understand it if i implemented it.

The first crack at this is using discrete inputs and outputs for letter-prediction (26 classes and space).
Inputs are 0/1 reals, and outputs are reals in [0,1] which attempt to learn one-hot 0/1 targets.


"""

import sys
import random
import torch
from numpy_rnn import NumpyRnn
import matplotlib.pyplot as plt
from torch_rnn import *
from torch_gru import *
from data_lib import *

def usage():
	print("Usage: python3 BPTT.py")
	print("Params (these apply differently to selected models): -eta,\
						-maxEpochs/-epochs, \
						-hiddenUnits,\
						-bpStepLimit,\
						-numSequences,\
						-miniBatchSize")
	print("Models: --torch-gru, --numpy-rnn, --custom-torch-rnn")

def main():
	eta = 1E-5
	hiddenUnits = 50
	maxEpochs = 500
	miniBatchSize = 2
	momentum = 1E-5
	bpStepLimit = 4
	maxSeqLen = 100
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
		if "-clip=" in arg:
			clip = float(arg.split("=")[-1])

	NUMPY_RNN = "--numpy-rnn"
	CUSTOM_TORCH_RNN = "--custom-torch-rnn"
	TORCH_GRU = "--torch-gru"

	validModels = {NUMPY_RNN, CUSTOM_TORCH_RNN, TORCH_GRU}
	if not any([arg in validModels for arg in sys.argv]):
		print("No model selected; must pass one of {}".format(validModels))
		usage()

	dataset, encodingMap = BuildCharSequenceDataset(limit=numSequences, maxSeqLen=20)
	reverseEncoding = dict([(encodingMap[key],key) for key in encodingMap.keys()])

	print("First few target outputs:")
	for sequence in dataset[0:20]:
		word = ""
		for x,y in sequence:
			index = np.argmax(y)
			word += reverseEncoding[index]
		print(word)

	print(str(encodingMap))
	#print(str(dataset[0]))
	print("SHAPE: num examples={} xdim={} ydim={}".format(len(dataset), dataset[0][0][0].shape, dataset[0][0][1].shape))
	xDim = dataset[0][0][0].shape[0]
	yDim = dataset[0][0][1].shape[0]

	#print("Shuffling dataset...")
	#random.shuffle(dataset)

	if "--numpy-rnn" in sys.argv:
		print("TODO: Implement sigmoid and tanh scaling to prevent over-saturation; see Simon Haykin's backprop implementation notes")
		print("TOOD: Implement training/test evaluation methods, beyond the cost function. Evaluate the probability of sequences in train/test data.")
		print("Building rnn with {} inputs, {} hidden units, {} outputs".format(xDim, hiddenUnits, yDim))
		net = NumpyRnn(eta, xDim, hiddenUnits, yDim, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH")
		#train the model
		net.Train(dataset, maxEpochs, miniBatchSize, bpStepLimit=bpStepLimit, clipGrad=clipGrad, momentum=momentum, saveMinWeights=saveMinWeights)
		print("Stochastic sampling: ")
		net.Generate(reverseEncoding, stochastic=True)
		print("Max sampling (expect cycles/repetition): ")
		net.Generate(reverseEncoding, stochastic=False)

	if "--custom-torch-rnn" in sys.argv:
		#convert the dataset to tensor form for pytorch
		dataset = convertToTensorData(dataset)
	
		rnn = DiscreteSymbolRNN(xDim, hiddenUnits, yDim)
		print("Training...")
		rnn.train(dataset, epochs=maxEpochs, batchSize=20, torchEta=eta, bpttStepLimit=bpStepLimit)
		rnn.generate(reverseEncoding)
	
	if "--torch-gru" in sys.argv:
		batchedData = convertToTensorBatchData(dataset, batchSize=miniBatchSize)
		#Try these params: python3 BPTT.py  -batchSize=4 -maxEpochs=6000 -momentum=0.9 -eta=1E-3
		gru = DiscreteGRU(xDim, hiddenUnits, yDim, numHiddenLayers=1, batchFirst=True, clip=clip)
		print("Training...")
		gru.train(batchedData, epochs=maxEpochs, batchSize=miniBatchSize, torchEta=eta)
		gru.generate(reverseEncoding,30,30,stochastic=True)
		gru.generate(reverseEncoding,30,30,stochastic=False)

if __name__ == "__main__":
	main()




