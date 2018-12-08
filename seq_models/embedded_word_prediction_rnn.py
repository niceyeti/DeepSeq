"""
A simple gru demonstration for discrete sequential prediction using pytorch. This is just for learning pytorch.
RNN model: Given an input symbol and the current hidden state, predict the next character. So we have
discrete one-hot input, and discrete one-hot output.

Sample output:
	
	python3 BPTT.py -batchSize=4 -maxEpochs=2000 -momentum=0.9 -eta=1E-3 -hiddenUnits=25
	...
	1649 2.7261360764503477
	1699 2.703751826286316
	1749 2.6933610081672668
	1799 2.730964684486389
	1849 2.6367283701896667
	1899 2.6822508335113526
	1949 2.603130030632019
	1999 2.6193048477172853
	Generating 10 sequences with stochastic=True
	^ztf^^e^^lzyt^q^zvfff^qvcjzxqmz<
	ydpzms^xsfbtjjjtjbpswdvpambawjq<
	ioovff^n^u^rt^wxamvbvzb^w^lpvwf<
	^npsbsu^zasaotaza^fjqvftamawpzz<
	ts^^lnwfawnuhaoszzpmzhpmzzzaovp<
	mfp^h^luhf^uoom^hp^zazppztpmfps<
	jjjectpfmauswjznvfieovbvatodmva<
	id bfmfuh$hpzmeoumjnptjjcpnpffz<
	wet^gv^t^y^xjljjbowfmvtit^hjxuw<
	kbqeouomoooooemgujeodoqvxzpz^da<
	Generating 10 sequences with stochastic=False
	zor wastherryound wasther$$$dly<
	ver and thering ther$$d$$d$$$d$<
	oun waskering$$drorky$$$$$d$$$$<
	noverry wasker$$d$$d$$$d$$$$d$$<
	ran her wastherr$$d$$d$$$$$$$$$<
	wand wastherryound$urker$$$d$$$<
	kand wastherring ther$$$d$$$$d$<
	ran her wastherr$$d$$d$$$$$$$$$<
	ghe wastherryound$urrery$$$d$$$<
	so waskering$doullyour$$$d$$$$$<


"""

import torch
import random
import matplotlib.pyplot as plt
from torch_optimizer_builder import OptimizerFactory

torch_default_dtype=torch.float32

#A GRU cell with softmax output off the hidden state; one-hot input/output, for a character prediction demo
#@useRNN: Using the built-in torch RNN is a simple swap, since it uses the same api as the GRU, so pass this to try an RNN
class EmbeddedGRU(torch.nn.Module):
	def __init__(self, xdim, hdim, ydim, numHiddenLayers, batchFirst, clip=-1, useRNN=False):
		super(EmbeddedGRU, self).__init__()
		
		self._optimizerBuilder = OptimizerFactory()
		self._batchFirst = batchFirst
		self.xdim = xdim
		self.hdim = hdim
		self.numHiddenLayers = numHiddenLayers
		#build the network architecture
		if not useRNN:
			self.gru = torch.nn.GRU(input_size=xdim, hidden_size=hdim, num_layers=numHiddenLayers, batch_first=self._batchFirst)
		else:
			self.gru = torch.nn.RNN(input_size=xdim, hidden_size=hdim, num_layers=numHiddenLayers, batch_first=self._batchFirst)
		self.linear = torch.nn.Linear(hdim, ydim)
		#LogSoftmax @dim refers to the dimension along which LogSoftmax (a function, not a layer) will apply softmax.
		# dim=2, since the output of the network is size (batchSize x seqLen x ydim) and we want to calculate softmax at each output, hence dimension 2.
		self.softmax = torch.nn.LogSoftmax(dim=2)
		self._initWeights()
		if clip > 0: #this is used as a flag to determine if clip_grad_norm_ will be called
			self._clip = 1
		else:
			self._clip = -1

	def _initWeights(self, initRange=1.0):
		#print("all: {}".format(self.gru.all_weights))
		for gruWeights in self.gru.all_weights:
			for weight in gruWeights:
				weight.data.uniform_(-initRange, initRange)
		self.linear.weight.data.uniform_(-initRange, initRange)

	def forward(self, x_t, hidden=None, verbose=False):
		"""
		@X_t: Input of size (batchSize x seqLen x xdim).
		@hidden: Hidden states of size (1 x batchSize x hdim), or None, which if passed will initialize hidden states to 0.

		Returns: @output of size (batchSize x seqLen x ydim), @z_t (new hidden state) of size (batchSize x seqLen x hdim)
		NOTE: Note that batchSize is in different locations of @hidden on input vs output
		"""
		z_t, hidden = self.gru(x_t, hidden) #@output contains all hidden states [1..t], whereas @hidden only contains the final hidden state
		s_t = self.linear(z_t)
		output = self.softmax(s_t)
		#print("x_t size: {}  z_t size: {} s_t size: {} output.size(): {} hidden: {}".format(x_t.size(), z_t.size(), s_t.size(), output.size(), hidden.size()))
		if verbose:
			print("x: {} hidden: {} z_t: {} s: {} output: {}".format(x_t, hidden, z_t, s_t, output))

		return output, z_t, hidden

	"""
	The axes semantics are (num_layers, minibatch_size, hidden_dim).
	Returns @batchSize copies of the zero vector as the initial state
	"""
	def initHidden(self, batchSize, numHiddenLayers=1, batchFirst=False, requiresGrad=True):
		if batchFirst:
			hidden = torch.zeros(batchSize, numHiddenLayers, self.hdim, requires_grad=requiresGrad)
		else:
			hidden = torch.zeros(numHiddenLayers, batchSize, self.hdim, requires_grad=requiresGrad)
		
		return hidden

	def initRandHidden(self, batchSize, numHiddenLayers=1, batchFirst=False, scale=1.0, requiresGrad=True):
		"""
		Initializes a random hidden state. This is for tasks like generation, from
		a random initial hidden state.

		@scale: Output of torch.randn contains numbers drawn from a zero mean 1-stdev Gaussian; @scale scales these to
		a different scale.
		"""
		if batchFirst:
			hidden = scale * torch.randn(batchSize, numHiddenLayers, self.hdim, requires_grad=requiresGrad)
		else:
			hidden = scale * torch.randn(numHiddenLayers, batchSize, self.hdim, requires_grad=requiresGrad)
		
		return hidden

	def sampleMaxIndex(self, v, stochastic=False):
		"""
		Given a stochastic vector (1 x n) vector @v, returns the max index of the vector
		under one of two strategies:
			@stochastic = false: just return the index of the max value in the vector (e.g. argmax(v))
			@stochastic = true: return the index sampled from the distribution of the vector. This
			is done by selecting a uniform random number in [0,1.0], then returning the index containing
			this number in its range. If v = [0.3, 0.3, 0.4] and r=0.5, returns 1, since the 0.5 occurs
			in the span of the second entry, [0.3-0.6).
		"""
		maxIndex = 0
		
		if not stochastic:
			#output of logsoftmax are log probabilities, so max prediction is max of output vector
			maxIndex = int(v.argmax(dim=0))
		else:
			#choose maxIndex stochastically according ot the distribution of the output
			p = torch.exp(v)  #get the output distribution as non-log probs
			r = torch.rand(1)[0]
			c = 0.0
			maxIndex = 0
			for i in range(p.size()[0]):
				if c >= r:
					maxIndex = i
					break
				c += p[i]

		return maxIndex

	def generate(self, reverseEncoding, numSeqs=1, seqLen=50, stochastic=False, allowRecurrentNoise=False):
		"""
		
		@reverseEncoding: A dict mapping integer indices to output strings, for reversing one-hot encodings back
						  into their domain representation (letters, words, etc).
		@numSeqs: Number of sequences to generate
		@seqLen: The length of each generated sequence, before stopping generation
		@stochastic: NOT IMPLEMENTED If True, then next character is sampled according to the distribution
					over output letters, as opposed to selecting the maximum probability prediction.
		@allowRecurrentNoise: Just an interesting parameter to observe: during generation, the output y'
		becomes the input to the next stage, and canonically should be one-hotted such that only the max
		entry is 1.0, and all others zero. However you can instead not one-hot the vector, leaving other noise
		in the vector. No idea what this will do, its just interesting to leave in.
		"""
		print("Generating {} sequences with stochastic={}".format(numSeqs,stochastic))

		for _ in range(numSeqs):
			#reset network
			hidden = self.initHidden(1, self.numHiddenLayers, requiresGrad=False)
			x_t = torch.zeros(1, 1, self.xdim, requires_grad=True)
			#print("hidden size {} x_0 size {}".format(hidden.size(), x_t.size()))
			maxIndex = random.randint(0,self.xdim-1)
			x_t[0][0][ maxIndex ] = 1.0
			s = reverseEncoding[maxIndex]
			for _ in range(seqLen):
				#@x_in output of size (1 x 1 x ydim), @z_t (new hidden state) of size (1 x 1 x hdim)
				x_t, z_t, hidden = self(x_t, hidden, verbose=False)
				#print("hidden size {} x_t size {}".format(hidden.size(), x_t.size()))
				maxIndex = self.sampleMaxIndex(x_t[0][0], stochastic)

				if not allowRecurrentNoise:
					x_t = x_t.zero_()
					x_t[0][0][maxIndex] = 1.0

				s += reverseEncoding[maxIndex]

			print(s+"<")

	def train(self, batchedData, epochs, batchSize=5, torchEta=1E-3, momentum=0.9, optimizer="sgd", ignoreIndex=-1):
		"""
		This is just a working example of a torch BPTT network; it is far from correct yet.
		The hyperparameters and training regime are not optimized or even verified, other than
		showing they work with the same performance as the rnn implemented in numpy from scratch.

		According to torch docs it might be possible to leave this is in its explicit example/update form,
		but instead simply accumulate the gradient updates over multiple time steps, or multiple sequences,
		by simply choosing when to zero the gradient with rnn.zero_grad().

		A very brief example from the torch docs, for reference wrt dimensions of input, hidden, output:
			>>> rnn = nn.GRU(10, 20, 2)    	  	# <-- |x|, |h|, num-layers
			>>> input = torch.randn(5, 3, 10) 	# <-- 1 batch of 5 training example in sequence of length 3, input dimension 10
			>>> h0 = torch.randn(2, 3, 20)		# <-- 2 hidden states matching sequence length of 3, hidden dimension 20; 2 hidden states, because this GRU has two layers
			>>> output, hn = rnn(input, h0)

		@dataset: A list of lists, where each list represents one training sequence and consists of (x,y) pairs
				  of one-hot encoded vectors.

		@epochs: Number of training epochs. Internally this is calculated as n/@batchSize, where n=|dataset|
		@batchSize: Number of sequences per batch to train over before backpropagating the sum gradients.
		@torchEta: Learning rate
		@bpttStepLimit: the number of timesteps over which to backpropagate before truncating; some papers are
				quite generous with this parameter (steps=30 or so), despite possibility of gradient issues.
		@ignore_index: Output target index values to ignore. These represent missing words or other non-targets. See pytorch docs.
		"""

		#define the negative log-likelihood loss function
		criterion = torch.nn.NLLLoss(ignore_index=ignoreIndex)
		#swap different optimizers per training regime
		optimizer = self._optimizerBuilder.GetOptimizer(parameters=self.parameters(), lr=torchEta, momentum=momentum, optimizer="adam")

		ct = 0
		k = 20
		losses = []
		nanDetected = False

		#try just allows user to press ctrl+c to interrupt training and observe his or her network at any point
		try:
			for epoch in range(epochs):
				x_batch, y_batch = batchedData[random.randint(0,len(batchedData)-1)]
				batchSeqLen = x_batch.size()[1]  #the padded length of each training sequence in this batch
				hidden = self.initHidden(batchSize, self.numHiddenLayers)
				# Forward pass: Compute predicted y by passing x to the model
				y_pred, z_pred, hidden = self(x_batch, hidden, verbose=False)
				# y_batch is size (@batchSize x seqLen x ydim). This gets the target indices (argmax of the output) at every timestep t.
				#batchTargets = y_batch.argmax(dim=1)
				print("Targets: {} {}".format(y_batch.size(), y_batch))
				#exit()
				# Compute and print loss. As a one-hot target nl-loss, the target parameter is a vector of indices representing the index
				# of the target value at each time step t.
				loss = criterion(y_pred, y_batch)
				nanDetected = nanDetected or torch.isnan(loss)
				losses.append(loss.item())
				if epoch % 50 == 49: #print loss eveyr 50 epochs
					avgLoss = sum(losses[epoch-k:])/float(k)
					print(epoch, avgLoss)
					if nanDetected:
						print("Nan loss detected; suggest mitigating with shorter training regimes (shorter sequences) or gradient clipping")
				#print(loss)
				# Zero gradients, perform a backward pass, and update the weights.
				optimizer.zero_grad()
				loss.backward()
				#note this is the wrong way to clip gradients, which should be done during backprop, not after backprop has accumulated all gradients, but torch doesn't support this easily
				if self._clip > 0.0:
				 	torch.nn.utils.clip_grad_norm_(self.parameters(), self._clip)
				optimizer.step()
	
		except (KeyboardInterrupt):
			pass

		#plot the losses
		k = 20
		avgLosses = [sum(losses[i:i+k])/float(k) for i in range(len(losses)-k)]
		xs = [i for i in range(len(avgLosses))]
		plt.plot(xs,avgLosses)
		plt.show()


