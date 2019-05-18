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

"""



"""
Make a simple batch-based rnn:
	input batch: (b x seq x xdim)
	output: (b x seq x nclasses)
"""















