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
    a = x.max(axis=1).reshape(x.shape[0], 1) # get max of each row, reshape result to match @x
    #print("a: "+str(a))
    z = x - a
    expZ = np.exp(z)
    #print("z: "+str(z))
    sumZ = np.sum(expZ, axis=1).reshape(x.shape[0], 1)
    #print("sum z: "+str(sumZ))
    return a + np.log(sumZ)

#@x: An m x n matrix
def stable_logsoftmax(x):
    return x - logsumexp(x)

#@x: An m x n matrix
def stable_softmax(x):
    a = x.max(axis=1).reshape(x.shape[0], 1) # get max of each row, reshape result to match @x
    #print("a: "+str(a))
    z = x - a
    #print("z: "+str(z))
    sumZ = np.sum(z, axis=1).reshape(x.shape[0], 1)
    #print("sum z: "+str(sumZ))
    expZ = np.exp(z)
    #print("expZ: "+str(expZ))
    return expZ / sumZ

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def concat_and_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)

### Define recurrent neural net #######

def create_rnn_params(input_size, state_size, output_size,
                      param_scale=0.01, rs=npr.RandomState(0)):
    return {'init hiddens': rs.randn(1, state_size) * param_scale,
            'change':       rs.randn(input_size + state_size + 1, state_size) * param_scale,
            'predict':      rs.randn(state_size + 1, output_size) * param_scale}

def rnn_predict(params, inputs):
    def update_rnn(input, hiddens):
        return np.tanh(concat_and_multiply(params['change'], input, hiddens))

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        #print("output shape: {}".format(output.shape))
        return stable_logsoftmax(output)
        #return output - logsumexp(output, axis=1, keepdims=True)     # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)
    output = [hiddens_to_output_probs(hiddens)]

    for input in inputs:  # Iterate over time steps.
        hiddens = update_rnn(input, hiddens)
        output.append(hiddens_to_output_probs(hiddens))
    return output

def rnn_log_likelihood(params, inputs, targets):
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
    return "".join([chr(np.argmax(c)) for c in one_hot_matrix])

def build_dataset(filename, sequence_length, alphabet_size, max_lines=-1):
    """
    - Loads a text file, and turns each line into an encoded sequence.
    - Each line is padded to @sequence_length with trailing space ' '
    - Each padded/fixed-length line is converted to a matrix (n x num_classes), where n is the length of the string and num_classes is the length of the one-hot encodings (number of output classes).
	- These matrices are loaded into @seqs, a matrix of size (n x num-lines x num_classes). Yes, annoyingly the examples are indexed via the center index ('ix', below).
	- Returns: @seqs a tensor of size ( x num-lines x num_classes)
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

    init_params = create_rnn_params(input_size=xDim, output_size=outputDim,
                                    state_size=hDim, param_scale=paramScale)

    def print_training_prediction(weights):
        print("Training text                         Predicted text")
        logprobs = np.asarray(rnn_predict(weights, train_inputs))
        for t in range(logprobs.shape[1]):
            training_text  = one_hot_to_string(train_inputs[:,t,:])
            predicted_text = one_hot_to_string(logprobs[:,t,:])
            print(training_text.replace('\n', ' ') + "|" +
                  predicted_text.replace('\n', ' '))

    def training_loss(params, iter):
        return -rnn_log_likelihood(params, train_inputs, train_inputs)

    def callback(weights, iter, gradient):
        if iter % 10 == 0:
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
