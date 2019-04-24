"""
This class is for embedded, sequential language models, and is intended to hide the details and mitigate the memory
consumption of very large training sets and word-vector models. Word2Vec/Fasttext models are typically 2-5Gb, and
training sequence files can be as big, so it pays to use generators/yield statements and such. This could even employ
caching to retain only the most frequent words' vectors in memory, or something similar.


The class accepts paths to:
	1) a word sequence file, where every line is a training sequence of pre-filtered/normalized text
	2) a word vector model (fasttext or word2vec)

Note the class is coupled to a entity-vector model.

This class places emphasis on clients to create training files with target properties. It does not randomize training data,
but could do so in the future by caching lines, or even by calling readline() a random number of times and discarding them.
It could also become quite slow (many disk reads) if batches are created slower than network training times.

"""

import torch
import vector_models
import os
import numpy as np

TORCH_DTYPE=torch.float32
NUMPY_DEFAULT_DTYPE=np.float32

#The index value to ignore in torch training. This value is used by torch as a label for output categories not to backprop.
IGNORE_INDEX = -1

class EmbeddedDataset(object):
	def __init__(self,						\
			trainPath, 						\
			modelPath,						\
			batchSize=3,					\
			#The number of batches to read-ahead and store
			batchCacheSize=100,				\
			torchDtype=TORCH_DTYPE,			\
			limit = -1,						\
			maxSeqLength = -1,				\
			minSeqLength = -1,				\
			useL2Norm = False):  #If true, use l2-norm of each word vector
		if not os.path.exists(trainPath):
			raise Exception("ERROR training text path not found: "+trainTextPath)
		if not os.path.exists(modelPath):
			raise Exception("ERROR word model path not found: "+wordModelPath)
		self._torchDtype = torchDtype
		#TODO: try True. This converts vectors to l2 norm. Might also just norm the entire model beforehand instead of re-norming every time the same word is looked up.
		self._useL2Norm = False
		self._maxSeqLength = maxSeqLength
		self._limit = limit
		self._minSeqLength = minSeqLength
		self._batchSize = batchSize
		self._batchCacheSize = batchCacheSize
		self._batchCache = []
		self._trainFile = open(trainPath, "r")
		self.Model = vector_models.loadModel(modelPath)
		#may not belong here, but works better instead of building these every time we read a training sample
		self._zero_vector_in = np.zeros(self.Model.layer1_size, dtype=NUMPY_DEFAULT_DTYPE) #vectors are stored by word2vec as (k,) size numpy arrays
		print("Built embedded dataset. Training file will be repeated, once examples are exhausted.")
		print("Consider passing useL2Norm to observe the effect of normalizing term vectors.")
		print("WARNING: not yet handling out-of-model terms!!! See @truncations in _getTrainingSequence")
		self.IgnoreIndex = IGNORE_INDEX

	def _getLine(self):
		line = self._trainFile.readline()
		#restart from beginning of file
		if line == "":	#readline returns empty str iff eof
			self._trainFile.seek(0)
			line = self._trainFile.readline()
		return line

	"""
	Generates batches of training sequences where each timestep prediction is from a k-dimensional embedding vector
	to the target index (int) of the next term, per its one-hot index in word2vec model. Only the target index
	of the target term is stored, not its one-hot encoded representation.

	TODO: This currently just skips any out of vocabulary term.

	Each training sequence is a sequence of tuples of this type, k \in R --> i \in Z+
	"""
	def getNextBatch(self):
		if len(self._batchCache) == 0:
			self._readAhead(self._batchCacheSize)
		return self._batchCache.pop()

	def _readAhead(self, n):
		#Reads @n batches into memory ahead of time. Not really much more than a cosmetic cache strategy as yet, since this the read cost is the same as reading one at a time.
		#But you could re-train on the same batches a few time, if of sufficient size, at least in early, steeper-gradient training phases.
		#@n: The number of batches to read-ahead
		self._batchCache = [self._getNextBatch() for i in range(n)]

	def _getNextBatch(self):
		batch = []
		while len(batch) < self._batchSize:
			trainingSeq = self._getTrainingSequence()
			if trainingSeq is not None:
				batch.append(trainingSeq)

		return self._batchifyTensorData(batch, self._batchSize, self.IgnoreIndex)

	def _getTrainingSequence(self):
		"""
		Reads single training sequence, reading lines until it constructs a sufficient example in terms of min/max length.
		Returns None if no example found.
		"""
		success = False
		retries = 0
		maxRetries = 100
		truncations = 0

		while not success and retries < maxRetries:
			line = self._getLine()
			seq = []
			#target outputs are stored via their corresponding model indices, not one-hot encoded
			outputs = []
			#note: min sequence length is checked later below, since due to omissions (words missing vectors) we don't immediately know the length of the sequence
			for word in line.split():
				if word in self.Model.wv.vocab:
					tensor = torch.tensor(self.Model.wv.word_vec(word, self._useL2Norm), dtype=self._torchDtype)
					seq.append(tensor)
					targetIndex = self.Model.wv.vocab[word].index
					outputs.append(targetIndex)
				else:
					truncations += 1
				if self._maxSeqLength > 0 and len(seq) > self._maxSeqLength:
					break
			trainingSeq = list(zip(seq+[self._zero_vector_in], [self.IgnoreIndex]+outputs))
			if self._isValidTrainingSequence(trainingSeq):
				success = True
			else:
				retries += 1

		if not success and retries >= maxRetries:
			print("WARNING maxRetries reached in dataset._getTrainingSequence")
			trainingSeq = None

		return trainingSeq

	def _isValidTrainingSequence(self, seq):
		#Returns true if @seq meets length requirements
		return (self._minSeqLength < 0 or \
			len(seq) >= self._minSeqLength) and \
			(self._maxSeqLength < 0 or \
			len(seq) <= self._maxSeqLength)

	def _batchifyTensorData(self, batch, batchSize=1, ignore_index=-1):
		"""
		The ugliest function, required by torch sequential batch-training models.
		"""
		#print("BATCH: "+str(batch))
		#print("BATCH SHAPE: {}".format(batch[0][0][0].shape))
		batches = []
		xdim = batch[0][0][0].shape[0]

		#convert to tensor data
		maxLength = max(len(seq) for seq in batch)
		batchX = torch.zeros(batchSize, maxLength, xdim).to(TORCH_DTYPE)
		batchY = torch.zeros(batchSize, maxLength).to(TORCH_DTYPE)
		batchY.fill_(ignore_index)

		for i, seq in enumerate(batch):
			for j, (x, y) in enumerate(seq):
				#batchX[i,j,:] = x.clone().detach().to(TORCH_DTYPE)
				#batchY[i,j] = y.clone().detach().to(TORCH_DTYPE)
				batchX[i,j,:] = torch.tensor(x).to(TORCH_DTYPE)
				batchY[i,j] = torch.tensor(y).to(TORCH_DTYPE)

		return batchX, batchY

