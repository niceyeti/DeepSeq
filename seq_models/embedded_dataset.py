"""
This class is for embedded, sequential language models, and is intended to hide the details and mitigate the memory
consumption of very large training sets and word-vector models. Word2Vec/Fasttext models are typically 2-5Gb, and
training sequence files can be as big, so it pays to use generators/yield statements and such. This could also employ
caching to retain only the most frequent words' vectors in memory, or something similar.


The class accepts paths to:
	1) a word sequence file, where every line is a training sequence of pre-filtered/normalized text
	2) a word vector model (fasttext or word2vec)

Note the class is coupled to an entity-vector model.

This class places emphasis on clients to create training files with target properties. It does not randomize training data,
but could do so in the future by caching lines, or even by calling readline() a random number of times and discarding them.
It could also become quite slow (many disk reads) if batches are created slower than network training times.
A better strategy is to randomize the input file, by shuffling or otherwise modifying it in advance to suit a purpose.

"""

import torch
import vector_models
import os
import numpy as np

TORCH_DTYPE=torch.float32
NUMPY_DEFAULT_DTYPE=np.float32

#The index value to ignore in torch training. This value is used by torch as a label for output categories not to backprop.
IGNORE_INDEX = -1

"""

"""
class EmbeddedDataset(object):
	def __init__(self,						\
			trainPath, 						\
			modelPath,						\
			batchSize=3,					\
			#The number of batches to read-ahead and store
			batchCacheSize=100,				\
			torchDtype=TORCH_DTYPE,			\
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
		self._minSeqLength = minSeqLength
		self._batchSize = batchSize
		self._batchCacheSize = batchCacheSize
		self._batchCache = []
		self._trainFile = open(trainPath, "r")
		self.Model = vector_models.loadModel(modelPath)
		#may not belong here, but works better instead of building these every time we read a training sample
		self._zero_vector_in = np.zeros(self.Model.layer1_size, dtype=NUMPY_DEFAULT_DTYPE) #vectors are stored by word2vec as (k,) size numpy arrays
		print("Built embedded dataset. Training file will be repeated, once examples are exhausted.")
		print(">>> Consider passing useL2Norm to observe the effect of normalizing term vectors.")
		print(">>> WARNING: not yet handling out-of-model terms!!! See @omissions in _getTrainingSequence")
		print(">>> These terms' vectors can be inferred, using...infer_vector()? Some word2vec model method that")
		print("I can't find, may be costly, and probably isn't well-supported by vector models as I'm creating them (not on KeyedVectors, not supported by FastText, etc)")
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
	to the target index (int) of the next term, its one-hot index in a word2vec model.

	TODO: This currently just skips any out of vocabulary terms.

	Each training sequence is a sequence of tuples of this type, k \in R --> i \in Z+
	"""
	def getNextPackedBatch(self):
		if len(self._batchCache) == 0:
			self._readAhead(self._batchCacheSize)
		return self._batchCache.pop()

	def _readAhead(self, n):
		"""
		Reads @n batches into memory ahead of time. Not really much more than a cosmetic cache strategy as yet, since this the read cost is the same as reading one at a time.
		But you could re-train on the same batches a few time, if of sufficient size, at least in early, steeper-gradient training phases.
		@n: The number of batches to read-ahead
		"""
		self._batchCache = [self._getNextBatch() for i in range(n)]

	def _getNextBatch(self):
		batch = []
		while len(batch) < self._batchSize:
			trainingTup = self._getTrainingSequence()
			if trainingTup is not None:
				batch.append(trainingTup)

		return self._batchifyTensorData(batch, self.IgnoreIndex)

	def _getTrainingSequence(self):
		"""
		Reads single training sequence, reading lines until it constructs a sufficient example in terms of min/max length.

		Returns: A training sequence as a list of tuples (x_t,y_t), where x_t is the input at time t and y_t is its
		target output as an integer, both taken from the vector model @Model. Returns none if no example found under length
		restrictions.
		"""
		success = False
		retries = 0
		maxRetries = 100

		while not success and retries < maxRetries:
			line = self._getLine()
			inputs = []
			omissions = 0
			#target outputs are stored via their corresponding model indices, not one-hot encoded
			outputs = []
			#note: min sequence length is checked later below, since due to omissions (words missing vectors) we don't immediately know the length of the sequence
			for word in line.split():
				if word in self.Model.wv.vocab:
					vec = self.Model.wv.word_vec(word, self._useL2Norm)
					inputs.append(vec)
					targetIndex = self.Model.wv.vocab[word].index
					outputs.append(targetIndex)
				else:
					omissions += 1
				if self._maxSeqLength > 0 and len(inputs) > self._maxSeqLength:
					break
			#Time align the predictions: x_t is input word vector time t, and its target, word at t+1
			#TODO: verify this, and also note its impact. Sometimes you want to backprop all the way to an initial state, and learn the prior term frequency/state distribution.
			inputs = inputs[:-1]
			outputs = outputs[1:]

			if self._isValidTrainingSequence(inputs, outputs, omissions):
				success = True
				#convert inputs into a single tensor of size (seqLen x xdim), omitting last input (last word is not an input) and first output (first word is not predicted, yet)
				x_tensor = torch.zeros(len(inputs), inputs[0].shape[0], dtype=self._torchDtype, requires_grad=False)
				for t in range(len(inputs)):
					x_tensor[t,:] = torch.from_numpy(inputs[t])
				trainingTup = (x_tensor, outputs)
			else:
				retries += 1

		if not success and retries >= maxRetries:
			print("WARNING maxRetries reached in dataset._getTrainingSequence")
			trainingTup = None

		return x_tensor, outputs

	def _isValidTrainingSequence(self, inputs, outputs, omissions):
		"""
		Returns true if sequence meets length requirements. Very simple for now. This function exists so I can
		tweek it for different datasets.
		@inputs: The sequence of k-dimensional input vectors
		@outputs: The integer sequence of target output classes
		@omissions: The number of word omissions (out of vocab words) encountered during the creation of this sequence
		"""
		return (self._minSeqLength <= 0 or \
			len(outputs)+1 >= self._minSeqLength) and \
			(self._maxSeqLength < 0 or \
			len(outputs)+1 <= self._maxSeqLength)

	def _batchifyTensorData(self, batch, ignore_index=-1):
		"""
		Converts @batch to a dataset of packed/padded sequences and target outputs.

		REMEMBER: If you pack+pad sequences, you need to unpack+unpad on the output of a network.

		@batch: A list of training sequences as returned by _getTrainingSequence().
		@ignore_index: An index of target outputs that will be ignored during backprop
		"""
		# training sequences must be sorted in descending length before padding/packing
		batch = sorted(batch, key=lambda tup: len(tup[1]), reverse=True) #TODO: Performance. insert training seqs in order instead of calling sorted()
		# get max length from the longest training sequence
		seqLen = len(batch[0][1])
		# get all input tensor sequences
		x_batch = [tup[0] for tup in batch]
		# get all outputs (class integers) and append ignore_index to all output sequences so they are all the same length. @ignore_index is ignored during training.
		y_batch = [yseq + [ignore_index for _ in range(seqLen-len(yseq))] for _, yseq in batch]
		y_batch = torch.LongTensor(y_batch)
		seqLens = [len(seq) for seq in x_batch]
		# pad the sequences
		paddedBatch = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True)
		# pack the sequences
		packedBatch = torch.nn.utils.rnn.pack_padded_sequence(paddedBatch, lengths=seqLens, batch_first=True)

		return packedBatch, y_batch

