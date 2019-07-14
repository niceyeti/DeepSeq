import numpy as np
import random
import matplotlib.pyplot as plt

numpy_default_dtype=np.float32

#Static helper class. All these functions are vector-valued.
class Neuron(object):
	@staticmethod
	def Tanh(z):
		return np.tanh(z)

	#NOTE: This assumes that z = tanh(x)!! That is, assumes z already represents the output of tanh.
	@staticmethod
	def TanhPrime(z_tanh):
		return 1 - z_tanh ** 2
		#return 1 - (Neuron.Tanh(z) ** 2)

	#@z: A vector. Softmax is a vector valued function. This is the numerically stable version of softmax
	@staticmethod
	def SoftMax(z):
		e_z = np.exp(z - np.max(z))
		return e_z / np.sum(e_z)

	@staticmethod
	def SoftMaxPrime(z):
		return 1.0

	@staticmethod
	def Sigmoid(z):
		return 1 / (1 + np.exp(-z))
		
	@staticmethod
	#NOTE: This assume @z_sig already represents a sigmoid output!
	def SigmoidPrime(z_sig):
		return z_sig * (1 - z_sig)

	#loss functions. These take in two vectors, y' and y*, and produce a scalar output.
	@staticmethod
	def SSELoss(y_prime, y_star):
		pass

	@staticmethod
	def SSELossDerivative(y_prime, y_star):
		pass

	@staticmethod
	def CrossEntropyLoss():
		pass

class NumpyRnn(object):
	"""
	@eta: learning rate
	@lossFunction: overall loss function, also setting its derivative function for training: XENT or SSE
	@outputActivation: Output layer function: tanh, softmax, linear
	@hiddenActivation: ditto
	@wShape: shape of the hidden-output layer matrix
	@vShape: shape of the input-hidden layer matrix
	@uShape: shape of the hidden-hidden layer matrix (the recurrent weights)
	"""
	#def __init__(self, eta=0.01, wShape, vShape, uShape, nOutputs, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH"):
	def __init__(self, eta, nInputs, nHiddenUnits, nOutputs, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH"):
		self._eta = eta
		self.SetLossFunction(lossFunction)
		self.SetOutputFunction(outputActivation)
		self.SetHiddenFunction(hiddenActivation)

		self.NumInputs = nInputs
		self.NumHiddenUnits = nHiddenUnits
		self.NumOutputs = nOutputs

		#Infer shape of weight matrices from input data model
		#The shapes are meant to be consistent with the following linear equations:
		#	V*x + U*s[t-1] + b_input = s[t]
		#	W*s + b_output = y
		vShape = (nHiddenUnits, nInputs)
		uShape = (nHiddenUnits, nHiddenUnits)
		wShape = (nOutputs, nHiddenUnits)   # W is shape (|y| x |s|)
		
		self._numInputs = nInputs
		self._numHiddenUnits = nHiddenUnits
		self._numOutputs = nOutputs

		#setup the parameters of a traditional rnn
		self.InitializeWeights(wShape, vShape, uShape, "random")

		#This is a gotcha, and is not well-defined yet. How is the initial state characterized, as an input? It acts as both input and parameter (to be learnt).
		#Clever solutions might include backpropagating one step prior to every training sequence to an initial input of uniform inputs (x = all ones), or similar hacks.
		#setup the initial state; note that this is learnt, and retained across predictions/training epochs, since it signifies the initial distribution before any input is received
		self._initialState = np.zeros(shape=(nHiddenUnits,1), dtype=numpy_default_dtype)

	def InitializeWeights(self, wShape, vShape, uShape, method="random"):
		if method == "random":
			self._W = np.random.rand(wShape[0], wShape[1]).astype(numpy_default_dtype)
			self._V = np.random.rand(vShape[0], vShape[1]).astype(numpy_default_dtype)
			self._U = np.random.rand(uShape[0], uShape[1]).astype(numpy_default_dtype)
		elif method == "zeros":
			self._W = np.zeros(shape=wShape, dtype=numpy_default_dtype)
			self._V = np.zeros(shape=vShape, dtype=numpy_default_dtype)
			self._U = np.zeros(shape=uShape, dtype=numpy_default_dtype)
		elif method == "ones":
			self._W = np.ones(shape=wShape, dtype=numpy_default_dtype)
			self._V = np.ones(shape=vShape, dtype=numpy_default_dtype)
			self._U = np.ones(shape=uShape, dtype=numpy_default_dtype)

		outputDim = wShape[0]
		hiddenDim = wShape[1] 
		#set the biases to vectors of ones
		self._outputBiases = np.ones(shape=(outputDim,1), dtype=numpy_default_dtype) #output layer biases; there are as many of these as output classes
		self._inputBiases  = np.ones(shape=(hiddenDim,1), dtype=numpy_default_dtype)

	def SetLossFunction(self, lossFunction):
		if lossFunction == "SSE":
			self._lossFunction = Neuron.SSELoss
			self._lossPrime = Neuron.SSELossDerivative
		elif lossFunction == "XENT":
			self._lossFunction = Neuron.CrossEntropyLoss
			self._lossPrime = Neuron.CrossEntropyLossDerivative

	"""
	Apparently its okay to drive an activation function, then softmax, thereby separating softmax from tanh/sigmoid one term,
	which is just weird. Sticking with a single output activation for now, where softmax can only be e^(x*w) / sum(e^x*w_i, i)
	"""
	def SetOutputFunction(self, outputFunction):
		if outputFunction == "TANH":
			self._outputFunction = Neuron.Tanh
			self._outputPrime = Neuron.TanhPrime
		elif outputFunction == "SIGMOID":
			self._outputFunction = Neuron.Sigmoid
			self._outputPrime = Neuron.SigmoidPrime
		elif outputFunction == "SOFTMAX":
			self._outputFunction = Neuron.SoftMax
			self._outputPrime = Neuron.SoftMaxPrime
		elif outputFunction == "LINEAR":
			self._outputFunction = Neuron.Linear
			self._outputPrime = Neuron.LinearPrime

	def SetHiddenFunction(self, hiddenFunction):
		if hiddenFunction == "TANH":
			self._hiddenFunction = Neuron.Tanh
			self._hiddenPrime = Neuron.TanhPrime
		elif hiddenFunction == "SIGMOID":
			self._hiddenFunction = Neuron.Sigmoid
			self._hiddenPrime = Neuron.SigmoidPrime
		elif hiddenFunction == "LINEAR":
			self._hiddenFunction = Neuron.Linear
			self._hiddenPrime = Neuron.LinearPrime

	"""
	Feed forward action of simple recurrent network. This function is stateful, before and after; after
	calling, the client is expected to read the output stored in self._Ys[-1].
	Post-condition: self._Ys[-1] contains the output prediction vector for x (the latest prediction),
	and the current hidden state is self._Ss[-1] and previous hidden state is self._Ss[-2].
	@x: A (|x|,1) input vector

	Note that this function implicitly takes as input the previous state, self._Ss[-1].
	"""
	def Predict(self, x):
		self._Xs.append(x)
		#get the (|s| x 1) state vector s
		s = self._V * x + self._U * self._Ss[-1] + self._inputBiases
		#drive signal through the non-linear activation function
		s = self._hiddenFunction(s)
		#save this hidden state
		self._Ss.append(s)
		#get the (|y| x 1) output vector
		y = self._W * s.T + self._outputBiases
		#drive the net signal through the non-linear activation function
		y = self._outputFunction(y)
		#save the output state
		self._Ys.append(y)

	"""
	Forward step of bptt entails setting the inputs of the entire network, and storing hidden states and outputs.

	@xs: A list of numpy vectors representing network inputs.

	Post-condition: self._Xs contains the entire sequence of inputs in @xs, 
	"""
	def ForwardPropagate(self, xs):
		self._Xs = []
		self._Ss = [self._initialState]
		self._Ys = []

		for x in xs:
			"""
			print("XDIM: {}".format(x.shape))
			print("VDIM: {}".format(self._V.shape))
			print("UDIM: {}".format(self._U.shape))
			print("WDIM: {}".format(self._W.shape))
			print("SDIM: {}".format(self._Ss[0].shape))
			print("INPUT BIASES: {}".format(self._inputBiases.shape))
			"""
			self._predict(x)

	#Stateful prediction: given current network state, make one output prediction
	def _predict(self,x):
		self._Xs.append(x)
		#get the (|s| x 1) state vector s
		s = self._V.dot(x) + self._U.dot(self._Ss[-1]) + self._inputBiases
		#drive signal through the non-linear activation function
		s = self._hiddenFunction(s)
		#save this hidden state
		self._Ss.append(s)
		#get the (|y| x 1) output vector
		y = self._W.dot(s) + self._outputBiases
		#drive the net signal through the non-linear activation function
		y = self._outputFunction(y)
		#save the output state; note that the output of the activation is saved, not the original input
		self._Ys.append(y)
		#print("YDIM: {}".format(y.shape))
		#print(str(y.T))
		return y

	#Returns column vector with one random bit high.
	def _getRandomOneHotVector(self,dim):
		r = np.random.randint(dim-1)
		return self._buildOneHotVector(dim, r)

	#Returns a colun vector with the chosen index high, all others zero
	def _buildOneHotVector(self, dim, i):
		v = np.zeros(shape=(dim,1))
		v[i,0] = 1.0
		return v

	def _selectStochasticIndex(self, yT):
		"""
		Given a horizontal (1xn) vector @yT of multinomial class probabilities, which by definition must sum to 1.0,
		and a number @r in [0.0,1.0], this returns the index of the class whose region @r falls within.
		This probabilistic choice procedure will choose the class with 0.8452... probability with
		probability 0.8452... by the central limit theorem.
		Precondition: @r is in [0.0,1.0] and sum(@yT) = 1.0.
		"""
		cdf = 0.0
		r = random.randint(0,1000) / 1000.0

		#print("r={} SHAPE: {}".format(r, yT.shape))
		for i in range(yT.shape[0]):
			cdf += yT[i][0]
			#print("cdf={} i={} r={}".format(cdf, i, r))
			if cdf >= r:
				#print("HIT cdf={} i={} r={}".format(cdf, i, r))
				return i

		return yT.shape[0]-1

	#Generates sequences by starting from a random state and making a prediction, then feeding these predictions back as input
	#@stochastic: If true, rather than argmax(y), the output is chosen probabilistically wrt each output class' probability.
	def Generate(self, reverseEncodingMap, stochastic=False):
		for i, c in reverseEncodingMap.items():
			y_hat = self._buildOneHotVector(self._numInputs, i)
			c = reverseEncodingMap[np.argmax(y_hat)]
			word = ""
			for i in range(20):
				word += c
				y_hat = self._predict(y_hat)
				#get the index of the output, either stochastically or just via argmax(y)
				if stochastic:
					y_i = self._selectStochasticIndex(y_hat)
					#print("{} sum: {}".format(y_hat, np.sum(y_hat)))
				else:
					y_i = np.argmax(y_hat)
				c = reverseEncodingMap[y_i]

			print(word)

	"""
	Utility for resetting network to its initial state. It still isn't clear what that initial
	state of the network should be; its a subtle gotcha missing from most lit.
	"""
	def _resetNetwork(self):
		self._Ss = [self._initialState]
		self._Xs = []
		self._Ys = []
		self._outputDeltas = []
		self._hiddenDeltas = []

	"""
	Given @y_target the target output vector, and @y_predicted the predicted output vector,
	returns the error vector. @y_target is y*, @y_predicted is y_hat.
	"""
	def GetOuptutError(self, y_target, y_predicted):
		#TODO: Map this to a specific loss function
		return y_target - y_predicted #SSE and softmax error

	def getMinibatch(self, dataset, k):
		"""
		Given a dataset as a sequence of (X,Y), select k examples at random and return as sequence.
		If @k >= len(dataset)/2, then the entire dataset is returned.
		"""
		n = len(dataset)
		if k > (n/2):
			return dataset

		examples = []
		for i in range(k):
			r_i = random.randint(0,n-1)
			examples.append(dataset[r_i])

		return examples

	"""
	For bpStepLimit, I implemented this using the kludgy solution of simply resetting the next (downstream) hidden layer gradient
	to the zero vector every bpStepLimit steps. Formally, truncated bptt backprops bpStepLimit steps from time t, accumulates these
	changes, then moves to step t-1 and backprops bpStepLimit steps. But notice the O(n^2) increase in training time. One can instead
	backprop bpStepLimit steps from time t, then simply reset the hidden layer gradient to the zero vector, effectively resetting
	backprop and continuing another bpStepLimit steps, which runs in linear time. This is still effective learning, since the expectation
	of learning examples doesn't change, and in fact it might encourage less overfitting per each sequence; but it is invalid from the
	perspective of forward propagating certain information but not backpropping it, since certain portions of a sequence will depend on that info,
	but won't backprop it.

	@saveMinWeights: If true, snapshot the weights at the minimum training error.
	"""
	def Train(self, dataset, maxEpochs=1000, miniBatchSize=4, bpStepLimit=4, clipGrad=False, momentum=0.0001, saveMinWeights=True, etaDecay=0.999, momentumDecay=0.999):
		losses = []
		dCdW = np.zeros(shape=self._W.shape, dtype=numpy_default_dtype)
		dCdV = np.zeros(shape=self._V.shape, dtype=numpy_default_dtype)
		dCdU = np.zeros(shape=self._U.shape, dtype=numpy_default_dtype)
		dCbI = np.zeros(shape=self._inputBiases.shape, dtype=numpy_default_dtype)
		dCbO = np.zeros(shape=self._outputBiases.shape, dtype=numpy_default_dtype)

		#momentum based deltas
		dCdW_prev = np.zeros(shape=self._W.shape, dtype=numpy_default_dtype)
		dCdV_prev = np.zeros(shape=self._V.shape, dtype=numpy_default_dtype)
		dCdU_prev = np.zeros(shape=self._U.shape, dtype=numpy_default_dtype)
		dCbI_prev = np.zeros(shape=self._inputBiases.shape, dtype=numpy_default_dtype)
		dCbO_prev = np.zeros(shape=self._outputBiases.shape, dtype=numpy_default_dtype)

		#under construction; for saving the weights at the minimum error during training (hackish, probably not worth it)
		W_min = np.zeros(shape=self._W.shape, dtype=numpy_default_dtype)
		V_min = np.zeros(shape=self._V.shape, dtype=numpy_default_dtype)
		U_min = np.zeros(shape=self._U.shape, dtype=numpy_default_dtype)
		Bi_min = np.zeros(shape=self._inputBiases.shape, dtype=numpy_default_dtype)
		Bo_min = np.zeros(shape=self._outputBiases.shape, dtype=numpy_default_dtype)

		count = 0
		random.shuffle(dataset)
		minLoss = 99999.0
		useMomentum = momentum > 0.0
		h_zeroes = np.zeros(shape=(self.NumHiddenUnits,1)) #memory optimization by not declaring new np matrices/vectors inside loops

		for _ in range(maxEpochs):
			#initialize the weight-change matrices in which to accumulate weight changes, since weights are tied in vanilla rnn's
			dCdW[:] = 0
			dCdV[:] = 0
			dCdU[:] = 0
			dCbI[:] = 0
			dCbO[:] = 0
			steps = 0

			miniBatch = self.getMinibatch(dataset, miniBatchSize)
			#accumulate gradients over all random sequences in mini-batch
			for sequence in miniBatch:
				count += 1
				if (count < 100 and count % 10 == 9) or count % 100 == 99:
					lastK = losses[max(0,count-99):count]
					avgLoss = sum(lastK) / len(lastK)
					if avgLoss < minLoss:
						minLoss = avgLoss
						if saveMinWeights:
							W_min = self._W[:]
							Bo_min = self._outputBiases[:]
							U_min = self._U[:]
							V_min = self._V[:]
							Bi_min = self._inputBiases[:]

					print("Example batch count {} avgLoss: {}  minLoss: {} eta: {:.3E} momentum: {:.3E}".format(count, avgLoss, minLoss, self._eta, momentum))
					#print("Example count {} avgLoss: {}  minLoss: {}  {}".format(count,avgLoss,minLoss, str(self._Ys[-1].T)))
				self._resetNetwork()

				#clipping the start/end of line characters input/outputs can be done here
				xs = [xyPair[0] for xyPair in sequence]
				ys = [xyPair[1] for xyPair in sequence]
				#forward propagate entire sequence, storing info needed for weight updates: outputs and states at each time step t
				self.ForwardPropagate(xs)
				#calculate all the hidden phi-primes (1-tanh**2)
				hiddenPrime = [self._hiddenPrime(s)	for s in self._Ss]

				#initialize the last hidden state (after output limit) to zero vector
				dhNext = h_zeroes
				bpSteps = 0
				for t in reversed(range(len(ys))):
				#for t in reversed(range(1,t_end)):
					#calculate output error at step t, from which to backprop
					y_target = sequence[t][1]
					e_output = y_target - self._Ys[t] #output error per softmax, |y| x 1 vector. In some lit, the actual error is (y^ - y*); but since we're descending this gradient, negated it is -1.0(y^-y*) = (y*-y^
					#cross-entropy loss. Only the correct output is included, by definition of cross-entropy: y* x log(y^); all correct 0 classes' terms are zero.
					#loss = np.sum(np.absolute(self._Ys[t] - y_target))
					loss = -np.log(self._Ys[t][np.argmax(y_target)])
					losses.append(loss)
					#W weight matrix can be updated immediately, from the output error
					dCdW += np.outer(e_output, self._Ss[t])
					#biases updated directly from e_output for output biases
					dCbO += e_output
					#get stationary output layer error wrt hidden layer
					dO = self._W.T.dot(e_output)
					#get the (recursive) hidden layer error wrt output layer and t+1 hidden layer error
					dH_t = dO
					if t < len(ys):
						dH_t += self._U.T.dot(dhNext) * hiddenPrime[t+1]

					if clipGrad:
						#clip the gradients (OPTIONAL)
						dH_t = np.clip(dH_t, -1.0, 1.0)

					#get the previous state; either t-1 state for t > 0, or the initial state distribution
					hPrev = self._Ss[t-1] if t > 0 else self._initialState
					#update the input and hidden weight matrices
					dCdU += np.outer(dH_t, hPrev) * hiddenPrime[t]
					dCdV += np.outer(dH_t, self._Xs[t]) * hiddenPrime[t]
					dCbI += dH_t
					bpSteps += 1
					if bpSteps > bpStepLimit:
						bpSteps = 0
						dhNext = h_zeroes
					else:
						dhNext = self._U.T.dot(dH_t)

			#apply the cumulative weight changes; the latter incorporates momentum
			if not useMomentum:
				self._W += self._eta * dCdW
				self._outputBiases += self._eta * dCbO
				self._U += self._eta * dCdU
				self._V += self._eta * dCdV
				self._inputBiases += self._eta * dCbI
			else:
				self._W += self._eta * dCdW + momentum * dCdW_prev
				self._outputBiases += self._eta * dCbO + momentum * dCbO_prev
				self._U += self._eta * dCdU + momentum * dCdU_prev
				self._V += self._eta * dCdV + momentum * dCdV_prev
				self._inputBiases += self._eta * dCbI + momentum * dCbI_prev
				dCdW_prev[:] = dCdW[:]
				dCbO_prev[:] = dCbO[:]
				dCdU_prev[:] = dCdU[:]
				dCdV_prev[:] = dCdV[:]
				dCbI_prev[:] = dCbI[:]

			if etaDecay > 0 and self._eta >= 5E-7: #shrink to a minimum of 5E-7
				self._eta *= etaDecay
			if momentumDecay > 0 and momentum >= 1E-7:
				momentum *= momentumDecay

		if saveMinWeights:
			#reload the weights from the min training error 			
			self._W = W_min[:]
			self._outputBiases = Bo_min[:]
			self._U = U_min[:]
			self._V = V_min[:]
			self._inputBiases = Bi_min[:]

		#plot the losses
		k = 500
		print("Plotting losses...")
		avgLoss = [sum(losses[i:i+k])/float(k) for i in range(0, len(losses)-k, k)]
		xs = [i for i in range(len(avgLoss))]
		plt.plot(xs, avgLoss)
		plt.show()

