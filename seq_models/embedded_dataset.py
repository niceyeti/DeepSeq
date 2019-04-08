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
TORCH_DTYPE=torch.float32
import vector_models

#The index value to ignore in torch training. This value is used by torch as a label for output categories not to backprop.
IGNORE_INDEX = -1

class EmbeddedDataset(object):
	def __init__(self,						\
			trainPath, 						\
			modelPath,						\
			batchSize=3,					\
			torchDtype=TORCH_DTYPE,			\
			limit = -1,						\
			maxSeqLength = -1,				\
			minSeqLength = -1,				\
			useL2Norm = False):  #If true, use l2-norm of each word vector
		if not os.path.exists(trainPath):
			except Exception("ERROR training text path not found: "+trainTextPath)
		if not os.path.exists(modelPath):
			except Exception("ERROR word model path not found: "+wordModelPath)
		self._torchDtype = torchDtype
		#TODO: try True. This converts vectors to l2 norm. Might also just norm the entire model beforehand instead of re-norming every time the same word is looked up.
		self._useL2Norm = False
		self._maxSeqLength = maxSeqLength
		self._limit = limit
		self._minSeqLength = minSeqLength
		self._trainFile = open(trainPath, "r")
		self.Model = vector_models.loadModel(modelPath)
		print("Built embedded dataset. Training file will be repeated, once examples are exhausted.")
		print("Consider passing useL2Norm to observe the effect of normalizing term vectors.")
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
		zero_vector_in = np.zeros(vecModel.layer1_size, dtype=self._torchDtype) #vectors are stored by word2vec as (k,) size numpy arrays
		truncations = 0
		batch = []

		for i in range(self._batchSize):
			line = self._getLine()
			seq = []
			#target outputs are stored via their corresponding model indices, not one-hot encoded
			outputs = []
			#note: min sequence length is checked later below, since due to omissions (words missing vectors) we don't immediately know the length of the sequence
			for word in line.split():
				if word in vecModel.vocab:
					tensor = torch.tensor(vecModel.word_vec(word, self._useL2Norm), dtype=self._torchDtype)
					seq.append(tensor)
					targetIndex = vecModel.vocab[word].index
					outputs.append(targetIndex)
				else:
					truncations += 1
				if self._maxSeqLength > 0 and len(seq) > self._maxSeqLength:
					break
			trainingSeq = list(zip(seq+[zero_vector_in], [ignore_index]+outputs))
			if len(trainingSeq) > minLength: #handles words missing vectors
				batch.append(trainingSeq)

		return self._batchifyTensorData(batch)

	def _batchifyTensorData(self, batch, batchSize=1, ignore_index=-1):
		"""
		The ugliest function, required by torch sequential batch-training models.
		"""

		batches = []
		xdim = batch[0][0].shape[0]

		#convert to tensor data
		maxLength = max(len(seq) for seq in batch)
		batchX = torch.zeros(batchSize, maxLength, xdim).to(TORCH_DTYPE)
		batchY = torch.zeros(batchSize, maxLength).to(TORCH_DTYPE)
		batchY.fill_(ignore_index)

		for i, seq in enumerate(batch):
			for j, (x, y) in enumerate(seq):
				batchX[i,j,:] = torch.tensor(x).to(TORCH_DTYPE)
				batchY[i,j] = torch.tensor(y).to(TORCH_DTYPE)

		return batchX, batchY

		"""
		print("X batch instance size (@batchSize x maxLength x xdim): {}".format(batches[0][0].size()))
		print("Y batch instance size (@batchSize x maxLength): {}".format(batches[0][1].size()))
		print("Ignore index (grads for these won't backprop): {}".format(ignore_index))

		return batches
		"""

	"""
	OBSOLETE
	Builds a sequential training dataset of word embeddings. Note that this could just implement a generator, rather
	than building a giant dataset in memory: the model contains all the vectors for the training sequences, so just
	iterate the training sequences, generating their corresponding vectors. 

	NOTE: If any term in @trainingText is not in the word2vec model, that training sequence is truncated at that word
	TODO: Instead of truncation, set unknown term to IGNORE_INDEX

	@trainTextPath: A file containing word sequences, one training sequence per line. The terms in this file must exactly match those
	used to train the word-embedding model. This is very important, since it means that both the model and the training sequences
	must have been derived from the same text normalization methods and so on.
	@wordModelPath: Path to a word2vec model for translating terms into fixed-length input vectors.
	@ignore_index: A flag value for output that should be ignored. See pytorch docs.

	Returns: Data as a list of sequences, each sequence a list of numpy one-hot encoded vectors indicating characters
	def loadEmbeddedDataset(limit=1000, maxSeqLen=1000000, minSeqLength=5, ignoreIndex=IGNORE_INDEX):
		if not os.path.exists(trainTextPath):
			print("ERROR training text path not found: "+trainTextPath)
			exit()
		if not os.path.exists(wordModelPath):
			print("ERROR word model path not found: "+wordModelPath)
			exit()

		#model = gensim.models.Word2Vec.load(wordModelPath)
		dataset = None
		trainFile = open(trainTextPath, "r")
		dataset = GetEmbeddedTrainingSequences(trainFile, model, minLength=minSeqLength, ignore_index=ignoreIndex)

		return dataset, model
	"""


