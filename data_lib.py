"""
A disorganized collection of helper functions for reading sequence data in different forms and formats.
"""

import string
import re
import numpy as np
import torch

#Best to stick with float; torch is more float32 friendly according to highly reliable online comments
numpy_default_dtype=np.float32

"""
Compresses all whitepace characters in some string down to a single-space.
"""
def CompressWhitespace(s):
	return " ".join(s.split()) #code golf ftw
	#s = s.replace("\r"," ").replace("\t"," ").replace("\n"," ")
	#s = s.replace("'","").replace("    "," ").replace("  "," ").replace("  "," ").replace("  "," ").replace("  "," ")
	#return s

#Strip non-alphabetic chars from a string, but including space char
def stripNonAlpha(s):
	return re.sub(r"[^a-zA-Z ]", '', s)

"""
Returns all words in some file, with all non-alphabetic characters removed, and lowercased.
"""
def GetSentenceSequence(fpath):
	words = []
	with open(fpath,"r") as ifile:
		#read entire character sequence of file
		novel = CompressWhitespace(ifile.read())
		sentences = [sentence.strip() for sentence in novel.split(".") if len(sentence.strip()) > 0]
		sentences = [stripNonAlpha(sentence).lower() for sentence in sentences]
		#lots of junk in the beginning, so toss it
		sentences = [sentence for sentence in sentences[100:] if sentence != " " and len(sentence) > 0]
		#print(novel)
	#print(sentences)

	return sentences

"""
Very very ad hoc: to achieve a maxSeqLen, just truncate chars after @maxSeqLen chars; e.g. s = s[0:maxLen].
This converts sentence data into a data often composed only of the beginning of sentences (depending on @maxLen),
which means a model will be fit to incomplete sequences. But training on shorter sequences prevents gradient explosion
in basic RNN models; gru/lstm tend not to need this as they're more stable.
"""
def TruncateSequences(sequences, maxLen):
	return [seq[0:maxLen] for seq in sequences]

"""
Returns a list of lists of (x,y) numpy vector pairs describing bigram character data: x=s[i], y=s[i-1] for some sequence s.

The data consists of character sequences derived from the novel Treasure Island.
Training sequences consist of the words of this novel, where the entire novel is lowercased,
punctuation is dropped, and word are tokenized via split(). Pretty simple--and inefficient. It will be neat to see 
what kind of words such a neural net could generate.

Each sequence consists of a list of numpy one-hot encoded column-vector (shape=(k,1)) pairs. The initial x in 
every sequence is the start-of-line character '^', and the last y in every sequence is the end-of line character '$'.
If this is undesired, these input/outputs can just be skipped in training; it is an unprincipled way of modeling
sequences begin/termination (see the Goodfellow Deep Learning book for better methods).

@limit: Number of sequences to return
@maxSeqLen: maximum length of any training sequence. This is important for simple RNN's, to prevent gradient explosion
by training over shorter sequences, e.g. @maxSeqLen=10 or so.
"""
def BuildCharSequenceDataset(fpath = "./data/treasureIsland.txt", limit=1000, maxSeqLen=1000000):
	dataset = []

	sequences = GetSentenceSequence(fpath)
	sequences = TruncateSequences(sequences, maxSeqLen)
	charMap = dict()
	i = 0
	for c in string.ascii_lowercase+' ':
		charMap[c] = i
		i+=1

	#add beginning and ending special characters to delimit beginning and end of sequences
	charMap['^'] = i
	charMap['$'] = i + 1
	print("num classes: {}  num sequences: {}".format(len(charMap.keys()), len(sequences)))
	numClasses = len(charMap.keys())
	startVector = np.zeros(shape=(numClasses,1), dtype=numpy_default_dtype)
	startVector[charMap['^'],0] = 1
	endVector = np.zeros(shape=(numClasses,1), dtype=numpy_default_dtype)
	endVector[charMap['$'],0] = 1
	for seq in sequences[0:limit]: #word sequence can be truncated, since full text might be explosive
		sequence = [startVector]
		#get the raw sequence of one-hot vectors representing characters
		for c in seq:
			vec = np.zeros(shape=(numClasses,1),dtype=numpy_default_dtype)
			vec[charMap[c],0] = 1
			sequence.append(vec)
		sequence.append(endVector)
		#since our input classes are same as outputs, just pair them off-by-one, such that the network learns bigram like distributions: given x-input char, y* is next char
		xs = sequence[:-1]
		ys = sequence[1:]
		sequence = list(zip(xs,ys))
		dataset.append(sequence)

	return dataset, charMap

"""
Converts a dataset, as returned by BuildSequenceData, to tensor form.
This is only suitable for the custom torch rnn, torch's builtin rnn, which expects tensor batches.
@dataset: A list of training sequences consisting of (x_t,y_t) vector pairs; the training sequences are also just python lists.
Returns: Returns the dataset in the same format as the input datset, just with the numpy vecs replaced by tensors.
"""
def convertToTensorData(dataset):
	print("Converting numpy data items to tensors...")
	dataset = [[(torch.from_numpy(x.T).to(torch.float32), torch.from_numpy(y.T).to(torch.float32)) for x,y in sequence] for sequence in dataset]
	return dataset

"""
Given a dataset as a list of training examples, each of which is a list of (x_t,y_t) numpy vector pairs,
converts the data to tensor batches of size @batchSize. Note that a constraint on batch data for torch rnn modules
is that the training sequences in each batch must be exactly the same length, when using the builtin rnn modules.
This may seem kludgy, but their api contains bidirectional models, so think about how you would batch train a bidirectional
model using graph-based computation if the sequences in the batch weren't the same length.

@dataset: A list of training examples, each of which is a list of (x,y) pairs, where x/y are numpy vectors
Returns: @batches, a list of (x,y) tensor pairs, each of which represents one training batch. x's are tensors of size
		(@batchSize x maxLength x xdim), and y's are tensors of size (@batchSize x maxLength x ydim)
"""
def convertToTensorBatchData(dataset, batchSize=1):
	batches = []
	xdim = dataset[0][0][0].shape[0]
	ydim = dataset[0][0][1].shape[0]

	print("Converting numpy data to tensor batches of padded sequences... TODO: incorporate pad_sequence() instead?")
	for step in range(0,len(dataset),batchSize):
		batch = dataset[step:step+batchSize]
		#convert to tensor data
		maxLength = max(len(seq) for seq in batch)
		batchX = torch.zeros(batchSize, maxLength, xdim)
		batchY = torch.zeros(batchSize, maxLength, ydim)
		for i, seq in enumerate(batch):
			for j, (x, y) in enumerate(seq):
				batchX[i,j,:] = torch.from_numpy(x.T).to(torch.float32)
				batchY[i,j,:] = torch.from_numpy(y.T).to(torch.float32)
		batches.append((batchX, batchY))
		#break
	print("Batch instance size: {}".format(batches[0][0].size()))

	return batches
