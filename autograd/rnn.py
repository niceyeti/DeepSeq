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
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
#from autograd.scipy.special import logsumexp
#from scipy.special import logsumexp
from os.path import dirname, join
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt

### Helper functions #################

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

def rnn_predict(params, inputs):
    """
    @params: params per create_rnn_params
    @inputs: batched inputs of size (seqlen x numExamples x numClasses). The order makes sense if you check out the for loop with update_rnn; this shape gives iteration over time steps.
    Returns: A list of output probabilities for the entire training set, whose entries are matrices of outputs for a single timestep.
    """
    def update_rnn(input, hiddens):
        """
        Calculate the hidden states, given the inputs and previous hidden states at this time step.
        @input: The stacked inputs at this time step.
        @hiddens: The hiddens at this timestep.
        """
        return np.tanh(concat_and_multiply(params['change'], input, hiddens))

    def hiddens_to_output_probs(hiddens):
        """
        Given a set of matrix of hidden states for a timestep (basically a vertical slice of the examples, or a column of hidden states),
        returns output probabilities for all of those states.
        @hiddens: The hidden states at timeslice t, of size (numExamples x hdim) where probs are over the hdim axis
        """
        output = concat_and_multiply(params['predict'], hiddens)
        #Hack fix for source example
        return stable_logsoftmax(output)
        #return output - logsumexp(output, axis=1, keepdims=True)     # Normalize log-probs.

    num_sequences = inputs.shape[1]
    # Initializes the (same) hidden state for every training sequence: size (numExamples x hdim)
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)
    #print("hiddens: {}".format(hiddens.shape))
    output = [hiddens_to_output_probs(hiddens)]

    # Iterate over time steps. In numpy, the for-iteration is over a tensor's first axis: 'for x in M', where M.shape=(4,6,7,8), would iterate 4 matrices of size (6,7,8)
    for input in inputs:
        hiddens = update_rnn(input, hiddens)
        output.append(hiddens_to_output_probs(hiddens))
    return output

def rnn_log_likelihood(params, inputs, targets):
    """
    @params: The params of the network
    @inputs: Batched training input of size (seqlen x numExamples x numClasses)
    @targets: Batched targets, which in this case are just the inputs.
    """
	#get the outputs for the entire batch, where @logprobs is size 
    logprobs = rnn_predict(params, inputs)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / (num_time_steps * num_examples)


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
####################################

if __name__ == '__main__':
    num_chars = 128
    losses = []
    seqLen = 30
    maxLines = 60
    xDim = 128
    outputDim = xDim
    hDim = 40
    paramScale = 0.01
    #training params
    numIters = 500
    adamStepSize = 0.1

    # Learn to predict our own source code.
    text_filename = join(dirname(__file__), __file__)
    train_inputs = build_dataset(text_filename, sequence_length=seqLen,
                                 alphabet_size=num_chars, max_lines=maxLines)

    init_params = create_rnn_params(xdim=xDim, odim=outputDim,
                                    hdim=hDim, param_scale=paramScale)
    print("Training data size: (seqlen x numexamples x classes) -> {}".format(train_inputs.shape))
    def print_training_prediction(weights):
        print("Training text                         Predicted text")
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
        return -rnn_log_likelihood(params, train_inputs, train_inputs)

    def callback(weights, iter, gradient):
        if iter % 10 == 0:
            print(type(iter))
            curLoss = training_loss(weights, 0)
            losses.append(curLoss)
            print("Iteration", iter, "Train loss:", curLoss)
            print_training_prediction(weights)

    # Build gradient of loss function using autograd.
    training_loss_grad = grad(training_loss)

    print("Training RNN...")
    trained_params = adam(training_loss_grad, init_params, step_size=adamStepSize,
                          num_iters=numIters, callback=callback)

    #plot the losses
    plt.plot([x for x in range(len(losses))], losses)
    plt.title("Losses")
    plt.show()

    print("\nGenerating text from RNN...")
    num_letters = 30
    for t in range(20):
        text = ""
        for i in range(num_letters):
            seqs = string_to_one_hot(text, num_chars)[:, np.newaxis, :]
            logprobs = rnn_predict(trained_params, seqs)[-1].ravel()
            text += chr(npr.choice(len(logprobs), p=np.exp(logprobs)))
        print(text)
