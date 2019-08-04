"""
All of this code was copied or derived directly from the autograd project, of Harvard university.
Used and modified under MIT license.

Implements the long-short term memory character model.
This version vectorizes over multiple examples, but each string
has a fixed length.



"""

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
import random
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
#from autograd.scipy.special import logsumexp
#from scipy.special import logsumexp
from os.path import dirname, join
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt

### Helper functions #################

def plotFoo(ys):
    #Plot any sequence of f(x)
    plt.plot([x for x in range(len(ys))], ys)
    plt.show()

def logsumexp(x):
    """
    Computes logsumexp for a matrix whose rows represent output probs. This is a hack to run the code,
    and would need to be refactored to support other input shapes.
    """
    a = x.max(axis=1).reshape(x.shape[0], 1) # get max of each row, reshape result to match @x
    #shift all rows down by their max
    z = x - a
    expZ = np.exp(z)
    sumZ = np.sum(expZ, axis=1).reshape(x.shape[0], 1)
    return a + np.log(sumZ)

def stable_logsoftmax(x):
    """
    Calculates logsoftmax over a set of output probabilities from one time slice of the training data.
    Or in English, given a matrix whose rows are output probabilities for a single time step, returns
    log-softmax of every row. See log-softmax lit for an explanation of how the math reduces to x - logsumexp(x).
    @x: An m x n matrix, m=numExamples n=numclasses. 
    """
    return x - logsumexp(x)

#@x: An m x n matrix
def stable_softmax(x):
    a = x.max(axis=1).reshape(x.shape[0], 1) # get max of each row, reshape result to match @x
    z = x - a
    sumZ = np.sum(z, axis=1).reshape(x.shape[0], 1)
    expZ = np.exp(z)
    return expZ / sumZ

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def concat_and_multiply(weights, *args): # *args is a variable param list, as a tuple.
    """
    Horizontally concatenates inputs (e.g., hiddens and x's, if needed), adds a vector of ones for the biases,
    then multiplies by the weights. Note that this works because our 'change' weights are the hidden/input
    matrices combined, which is a code idiom often used in vanilla rnn's. Hence their dot-product is
    the same.

    @weights: A weight matrix of size (m x n)
    @args: A variable-length list of args. For output prediction, this is the hidden states for a time slice.
    For input+hidden state updates, this is (input, hiddens).
    """
    #cats a set of constant ones to the inputs (args) to backpropagate the biases
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    #print("Cat shape: {} {}\n weights: {} {} \n args: {} {}\n".format(cat_state.shape, cat_state, weights.shape, weights, args[0].shape, args[0]))
    return np.dot(cat_state, weights)

### Define recurrent neural net #######

def create_rnn_params(xdim, hdim, odim,
                      param_scale=0.01, rs=npr.RandomState(0)):
    return {'init hiddens': rs.randn(1, hdim) * param_scale,
            # W_hh and W_xh, concatenated
            'change':       rs.randn(xdim + hdim + 1, hdim) * param_scale,
            # W_hy
            'predict':      rs.randn(hdim + 1, odim) * param_scale}

def rnn_predict(params, inputs, hidden=None):
    """
    Calculates and returns the outputs for inputs.shape[0] timesteps: At each stage, feed in the prior hidden state and current input.
    Usage for generator-mode: pass in batched @inputs of shape (1 x numExamples x numChannels), so length-one sequences. Get the output, and feed this back in as input by calling rnn_predict again.

    NOTE: This uses some goofy hacks and idioms since this is a discrete step model of infinite real-valued sequences.
        1) No START idiom, since the training domain represents infinite sequences (likewise no END terminal symbol either)
        2) This means mapping the inputs to themselves but offset by t+1, which means losing at least one data point at the end.
        3) This means the code is coupled to the same idioms being used in rnn_l2_loss, which is where the t+1 offset is visible.
    Also this api was taken from a single-purpose example, hence its restricted semantics for training vs. generation/prediction.

    @params: params per create_rnn_params
    @inputs: batched inputs of size (seqlen x numExamples x numChannels). The order makes sense if you check out the for loop with update_rnn; this shape gives iteration over time steps.
    @hiddens: Hidden states of size (numExamples x hdim). Pass these when generating predictions; not when training.
    Returns: A list of matrices of size (numExamples x numClasses), where the length of the list is the number of timesteps.
    """
    def update_rnn(input, hiddens):
        """
        Calculate the hidden states, given the batch inputs and previous hidden states at this time step.
        @input: The stacked inputs at this time step, shape (batchSize x numChannels) "numExamples" aka batchSize
        @hiddens: The hidden states at this timestep, shape (batchSize x hdim)
        """
        return np.tanh(concat_and_multiply(params['change'], input, hiddens))

    def hidden_to_output(hidden):
        """
        Given a set of matrix of hidden states for a timestep (basically a vertical slice of the examples, a column of hidden states for a specific time step),
        returns output probabilities for all of those states.
        @hiddens: The hidden states at timeslice t, of size (numExamples x hdim). Output probs are over the hdim axis. "numExamples" aka batchSize
        Returns: batch output matrix for one timestep of size (numExamples x numClasses)
        """
        output = concat_and_multiply(params['predict'], hidden)
        return output
        #Hack fix for source example
        #return stable_logsoftmax(output)
        #return output - logsumexp(output, axis=1, keepdims=True)     # Normalize log-probs.

    num_sequences = inputs.shape[1]
    if hidden is None:
        # Initializes the (same) hidden state for every training sequence: size (numExamples x hdim)
        hidden = np.repeat(params['init hiddens'], num_sequences, axis=0)
        #TODO: How to initialize initial hidden states, given infinite sequence model. Maybe the solution is just to increase training length, and similar training data hacks.
        #hiddens[:] = 0

    #print("hiddens: {}".format(hiddens.shape))
    #outputs = [hiddens_to_output(hiddens)]
    outputs = []
    hiddens = [hidden]
    # Iterate over time steps. In numpy, the for-iteration is over a tensor's first axis: 'for x in M', where M.shape=(4,6,7,8), would iterate 4 matrices of size (6,7,8)
    for input in inputs:
        hidden = update_rnn(input, hidden)
        hiddens.append(hidden)
        outputs.append(hidden_to_output(hidden))

    return outputs, hiddens

def rnn_l2_loss(params, inputs, targets):
    """
    For a batch of training sequences, returns the loss over the entire set, and for all time steps.
    The loss if the log-likelihood averaged over all training examples (time steps x num-sequences).
    @params: The params of the network
    @inputs: Batched training input of size (seqlen x numExamples x numClasses)
    @targets: Batched targets, which in this case are just the inputs.
    """
	#get the outputs for the entire batch, where @logprobs is size 
    preds, _ = rnn_predict(params, inputs)
    loss = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    # num_time_steps - 1 AND targets[t+1] used here since we are trying to map inputs to themselves, offset by one step. Note this means losing one data point.
    for t in range(num_time_steps-1):
        loss += np.sum( (preds[t] - targets[t+1]) ** 2 )
    return loss / (num_time_steps * num_examples)

### Dataset setup ##################
def string_to_one_hot(string, num_classes):
    """
    Converts an ASCII string to a one-of-k encoding.
    Returns: an array of one-hot encoded vectors of size (n x num_classes) where n = len(string) = seqLen, and num_classes is length of one-hot vecs.
    The numpy idioms are very dense here, just take them one at a time.
    """
	# @ascii is shape (k,) where k = len(string)
    ascii = np.array([ord(c) for c in string]).T
	# 1) np.arange(3) is array([0, 1, 2]).  np.arange(3) is array([[0, 1, 2]]), and so on. Each 'None,' adds another axis [] to the tensor: shape (3,), (1,3), (1,1,3) and so on.
    #    So the result on the rhs below is shape (1,128)
	# 2) lhs of '==' is shape (k x 1) (column vector of string-char ords), rhs is (1 x num_classes).
	# 3) The expression '==' of a column vec and a row vec is an idiom which generates one-hot encoded vectors. Example: np.array([1,2,3])[:,None] == np.array([1,2,3])[None,:]
    return np.array(ascii[:,None] == np.arange(num_classes)[None, :], dtype=int)

def one_hot_to_string(one_hot_matrix):
    #Convert a matrix of logsoftmax output probs (or one-hots) to their corresponding char sequence, per each row max
    return "".join([chr(np.argmax(c)) for c in one_hot_matrix])

def build_dataset(filename, sequence_length, alphabet_size, max_lines=-1):
    """
    - Loads a text file, and turns each line into an encoded sequence.
    - Each line is padded to @sequence_length with trailing space ' '
    - Each padded/fixed-length line is converted to a matrix (n x num_classes), where n is the length of the string and num_classes is the length of the one-hot encodings (number of output classes).
	- These matrices are loaded into @seqs, a matrix of size (n x num-lines x num_classes). Yes, annoyingly the examples are indexed via the center index ('ix', below).
	- Returns: @seqs a tensor of size (seqlen x num-lines x num_classes)
	"""
    with open(filename) as f:
        content = f.readlines()
    content = content[:max_lines]
    content = [line for line in content if len(line) > 2]   # Remove blank lines
    seqs = np.zeros((sequence_length, len(content), alphabet_size))
    for ix, line in enumerate(content):
        padded_line = (line + " " * sequence_length)[:sequence_length]
        seqs[:, ix, :] = string_to_one_hot(padded_line, alphabet_size)
    return seqs

#Returns list of points along a sinusoid, given the parameters.
def get_sinusoid(num_data_points, points_per_period, phase_shift):
    return [np.sin(i * (2*np.pi / points_per_period) + phase_shift) for i in range(num_data_points)]

def build_sinusoidal_dataset(sequence_length, points_per_period, output_channels, batch_size=30):
    """
    Returns a dataset of size (seqlen x batch_size x num_channels), where each training sequence
    is a sine function phase shifted by some random number of radians. The unusual matrix shape is
    a code idiom often used in rnn batching/training.
    FUTURE: Multichannel, multiple functions: channel 1 = sine, channel 2 = cosine.
    FUTURE: Gaussian noise
    @sequence_length: The total number of data points in each sinusoid. Cannot be less than points_per_period
    @points_per_period: The number of data points within each complete sinusoid.
    Therefore the approximate number of complete waves per example is sequence_length / wave_steps.
    @output_channels: Not used yet, set to one. This would be the number of outputs to predict over.
    @batch_size: The number of sequences to include.
    """

    if sequence_length < points_per_period:
        raise Exception("@sequence_length={} cannot be less than @points_per_period={}".format(sequence_length, points_per_period))

    #Each sequence is one complete sine wave (2Pi) scaled to fit within @sequence_length in discrete steps
    data = np.zeros((sequence_length, batch_size, output_channels))
    for b in range(batch_size):
        phaseShift = random.random() * 2 * np.pi
        #wave = [np.sin(i * (2*np.pi / points_per_period) + phaseShift) for i in range(sequence_length)]
        wave = get_sinusoid(sequence_length, points_per_period, phaseShift)
        #FUTURE: multichannel input/output. Just append more waves: [wave1, wave2, wave2...]
        train = np.array([wave])
        #print("train.T: {} {}".format(train.T.shape, train.T))
        #channels = train.T.reshape((sequence_length, 1, output_channels))
        #channels = np.array([wave,wave2]).reshape(sequence_length,1,output_channels)
        #print("Channels shape: {}\n channels {}".format(channels.shape, channels))
        #plotFoo(wave)
        data[:,b,:] = train.T
        #print("data[:,{},:]: {}".format(i,data[:,i,:]))

    #print("data[:,0,:]: {}".format(data[:,0,:]))
    return data

####################################

if __name__ == '__main__':
    losses = []
    #data points per complete sinusoid period
    points_per_period = 30
    seqLen = 1 * points_per_period
    batchSize = points_per_period
    xDim = 1
    outputChannels = xDim
    hDim = 16
    paramScale = 0.01
    #training params
    numIters = 250
    adamStepSize = 0.1

    # Learn to predict our own source code.
    train_inputs = build_sinusoidal_dataset(sequence_length=seqLen, points_per_period=points_per_period, output_channels=1, batch_size=batchSize)
    print("Training shape: {}".format(train_inputs.shape))
    init_params = create_rnn_params(xdim=xDim, odim=outputChannels, hdim=hDim, param_scale=paramScale)
    print("Training data size: (seqlen x numexamples x output channels) -> {}".format(train_inputs.shape))
    def print_training_prediction(weights):
        print("Training                         Predicted")
        logprobs = np.asarray(rnn_predict(weights, train_inputs))
        for t in range(logprobs.shape[1]):
            training_text  = one_hot_to_string(train_inputs[:,t,:])
            predicted_text = one_hot_to_string(logprobs[:,t,:])
            print(training_text.replace('\n', ' ') + "|" + predicted_text.replace('\n', ' '))

    def training_loss(params, iter):
        """
        @params: The model parameters, per create_rnn_params
        @iter: ?
        """
        return rnn_l2_loss(params, train_inputs, train_inputs)

    def callback(weights, iter, gradient):
        if iter % 10 == 0:
            curLoss = training_loss(weights, 0)
            losses.append(curLoss)
            print("Iteration", iter, "Train loss:", curLoss)
            #print_training_prediction(weights)

    # Build gradient of loss function using autograd.
    training_loss_grad = grad(training_loss)

    print("Training RNN...")
    trained_params = adam(training_loss_grad, init_params, step_size=adamStepSize, num_iters=numIters, callback=callback)

    #plot the losses
    plt.plot([x for x in range(len(losses))], losses)
    plt.title("Losses")
    plt.show()

    print("\nGenerating signals from RNN...")
    num_signals = 1
    num_steps = seqLen * 2
    #Initialize some random starting points, [0,2Pi], of shape (1 x numExamples x numChannels) where 1 == seqLen during generation
    inputs = np.random.random((1,num_signals,1)) * 2 - 1
    hidden = np.random.random((num_signals,hDim)) * 0.2  - 0.1  #(numExamples x hdim)
    print("In: {}".format(inputs))
    outputs = [inputs.reshape(num_signals, outputChannels)] # A list of output matrices of size (numSignals x numChannels)
    print("Inputs: {}".format(inputs))
    for _ in range(num_steps):
        rnn_outputs, hiddens = rnn_predict(trained_params, inputs, hidden)
        out_batch = rnn_outputs[0]
        hidden = hiddens[-1]
        print("In: {}\nOut: {}".format(inputs, out_batch))
        outputs.append(out_batch)
        inputs = out_batch.reshape((1, num_signals, xDim)) # out_batch is (numSignals x numChannels); convert to input size (1 x numExamples/batchSize x numChannels)
 
    print("Generated outputs: {}".format(outputs))
    #Plot a few channels
    sigNumber=0
    channel=0
    sig1 = [output[sigNumber, channel] for output in outputs]
    plotFoo(sig1)

    """
    print("\nGenerating signals from RNN...")
    num_signals = 2
    for t in range(seqLen):
        text = ""
        for i in range(num_letters):
            seqs = string_to_one_hot(text, num_chars)[:, np.newaxis, :]
            logprobs = rnn_predict(trained_params, seqs)[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print(text)
    """
    
