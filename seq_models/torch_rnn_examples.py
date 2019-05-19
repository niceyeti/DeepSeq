"""
Key:
	'b' : batch size
	'seq': training sequence length
	'xdim': input dimension real-valued x vectors
	'ydim': num output classes

The generic torch optimization pattern:

	optim.SGD([  {'params': model.base.parameters()},
		         {'params': model.classifier.parameters(), 'lr': 1e-3}
		      ], lr=1e-2, momentum=0.9)

	for input, target in dataset:
		optimizer.zero_grad()
		output = model(input)
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()

This provides several examples of:
	1) Packed batches
	2) Simple rnn layers, their expected inputs to forward(), and expected outputs using packed batches

Using packed batches is fairly confusing, but the reasons for using them is that they can avert many computations
for effectively the same training, but at lower computational costs. This is probably more important for gpu
based training, which I don't use and for which this example does not provide a good example (sorry).


"""

import torch




def getBatch():
	"""
	Torch (input) batches are in the shape (b x seq x xdim), where b is the batchsize, seq is the sequence length of the training
	examples, and xdim is the dimension of each x input at a particular timestep. So (5 x 20 x 300) would be a batch of
	5 training examples of length 20 with an input dimension of 300 (e.g., these could be vector embedding inputs).
	"""

	"""
	Here, read some data and convert to training sequences... I am intentionally using program variables gratuitously
	here to identify the role of these numbers, which are often just magic numbers in the abundant of unhelpful online examples.
	"""
	# 'ts[x]' = 'training sequence x', each with different lengths. I'm using program variables gratuitously here to identify these 
	xdim = 3 #input (embedding) vector dimension
	length_1 = 2	#length of this training sequence
	ts1 = torch.rand(length_1, xdim)
	# Now make ts2, an example of length 8, ts3 of length 7, etc...
	length_2 = 4
	ts2 = torch.rand(length_2, xdim)
	length_3 = 5
	ts3 = torch.rand(length_3, xdim)

	seqBatch = sorted([ts1, ts2, ts3], key=lambda seq: len(seq), reverse=True)
	seqLens = [len(seq) for seq in seqBatch]
	print("Seq batch:\n",seqBatch)
	#pad the sequences
	paddedBatch = torch.nn.utils.rnn.pad_sequence(seqBatch, batch_first=True)
	print("Padded batch:\n", paddedBatch)
	packedBatch = torch.nn.utils.rnn.pack_padded_sequence(paddedBatch, lengths=seqLens, batch_first=True)
	print("Packed batch:\n", packedBatch)

	return packedBatch

def trainingExamples(packedBatch):
	"""
	Por gratia... a few simple lstm and gru layer examples, using a packed batch as output by getBatch().
	"""
	xdim = 3 #Note that these dimension values are duplicated in getBatch(). These are just here for clarity, otherwise you would want to maintain them in a single locationin your code.
	hdim = 3
	batchSize = 3
	seqLen = 5
	numLayers=1
	numDirections = 1

	#build an lstm layer of the same xdim as hard-coded into the getBatch() batches. @hidden_size not relevant here, just arbitrarily 3.
	lstm = torch.nn.LSTM(input_size=xdim, hidden_size=hdim, batch_first=True)
	"""
	Notes per next line:
		@output: Using a packed batch, 
		@hn: Hidden state for initializing each training sequence, of size num-layers x batch-size x hdim.
		@cn: Ditto, but the cell state.
	"""
	h0 = torch.rand(numLayers, batchSize, hdim) #initial hidden state: (num-layers*num-directions  x  batch-size  x  hdim)
	c0 = torch.rand(numLayers, batchSize, hdim) #initial cell state:   (num-layers*num-directions  x  batch-size  x  hdim)
	output, (hn, cn) = lstm(packedBatch, (h0, c0))
	print("Output:\n", output)
	print("hn:\n", hn, hn.size())
	print("cn:\n", cn, cn.size())
	#Now run the output through the reverse of pack_padded_sequence to undo the packing operation.
	paddedOutput, outputLens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=seqLen)
	print("Padded output:\n", paddedOutput, paddedOutput.size())
	print("outputLens:\n", outputLens, outputLens.size())

	#Now a demo of a GRU layer, just to highlight its differences with LSTM (btw, torch's vanilla RNN layer has the same I/O this GRU)
	gru = torch.nn.GRU(input_size=xdim, hidden_size=hdim, num_layers=numLayers, batch_first=True)
	"""
	Notes per next line:
		Note the gru doesn't have a cn state, unlike the lstm, giving it a simpler interface.
	"""
	h0 = torch.rand(numLayers, batchSize, hdim) # Tensor of initial hidden states
	# @output: Size (  x  seq_len  x  num_directions * hidden_size). The outputs of the GRU (hidden states) across all sequences, timesteps.
	# @hn: Size (1  x  batch-size  x hdim). Tensor containing the hidden states over all training sequences for t = seq_len
	output, hn  = gru(packedBatch, h0)
	print("Output:\n", output)
	print("hn:\n", hn, hn.size())
	#Now run the output through the reverse of pack_padded_sequence to undo the packing operation.
	paddedOutput, outputLens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=seqLen)
	print("Padded output:\n", paddedOutput, paddedOutput.size())
	print("outputLens:\n", outputLens, outputLens.size())

	"""
	TODO: a bidirectional GRU...
	"""






"""
Make a simple batch-based rnn:
	input batch: (b x seq x xdim)
	output: (b x seq x nclasses)
"""

batch = getBatch()

trainingExamples(batch)













