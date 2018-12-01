"""
Simple builder class for getting a torch.nn.optimizer object, which is technically a hyperparameter of
gradient based methods. This was factored out since it can be used by many different nn modules, instead
of each one implementing its own. Could be just a singleton or static class.

This isn't well-factored and isn't production code; just needed a quick builder class.
"""

import torch.optim

class OptimizerFactory(object):
	def __init__(self):
		self._validOptimizers = {
									"sgd", \
									"adadelta", \
									"adagrad", \
									"adam", \
									"sparseadam", \
									"adamax", \
									"asgd", \
									"lbfgs" \
								}

	def GetOptimizer(self, **kwargs):
		"""
		Big ugly parameters list; they are not all-inclusive, some optimizers use different subsets of the
		parameters but not others:
			parameters: parameters of your nn model (torch.nn module)
			optimizer: string in self._validOptimizers indicating optimizer
			lr: learning rate in 0.0-1.0, but usually [1E-1, 1E-6], and sometimes changes dynamically if the optimizer 'says so'
			momentum: momentum value in 0.0-1.0
			Others: max_iter, max_eval, 

		"""

		params = kwargs["parameters"]
		optimizer = kwargs["optimizer"].lower()
		lr = kwargs["lr"]
		if "optimizer" not in kwargs:
			print("No optimizer passed, defaulting to SGD")
			optimizer = "sgd"
		if optimizer not in self._validOptimizers:
			print("Optimizer {} not in valid optimizers; defaulting to sgd".format(self._validOptimizers))
			optimizer = "sgd"

		momentum = None
		rho = None
		eps = None
		weight_decay = None
		lr_decay = None
		initial_accumulator_value = None
		betas = None
		lambd = None
		alpha = None
		t0 = None
		tolerance_grad = None
		tolerance_change = None
		history_size = None
		line_search_fn = None
		step_sizes = None
		dampening = None
		nesterov = None
		amsgrad = None

		if "momentum" in kwargs:
			momentum = kwargs["momentum"]
		if "rho" in kwargs:
			rho = kwargs["rho"]
		if "eps" in kwargs:
			eps = kwargs["eps"]
		if "weight_decay"  in kwargs:
			weight_decay = kwargs["weight_decay"]
		if "lr_decay" in kwargs:
			lr_decay = kwargs["lr_decay"]
		if "initial_accumulator_value" in kwargs:
			initial_accumulator_value = kwargs["initial_accumulator_value"]
		if "betas" in kwargs:
			betas = kwargs["betas"]
		if "lambd" in kwargs:
			lambd = kwargs["lambd"]
		if "alpha" in kwargs:
			alpha = kwargs["alpha"]
		if "t0" in kwargs:
			t0 = kwargs["t0"]
		if "tolerance_grad" in kwargs:
			tolerance_grad = kwargs["tolerance_grad"]
		if "tolerance_change" in kwargs:
			tolerance_change = kwargs["tolerance_change"]
		if "history_size" in kwargs:
			history_size = kwargs["history_size"]
		if "line_search_fn" in kwargs:
			line_search_fn = kwargs["line_search_fn"]
		if "step_sizes" in kwargs:
			step_sizes = kwargs["step_sizes"]
		if "dampening" in kwargs:
			dampening = kwargs["dampening"]
		if "nesterov" in kwargs:
			nesterov = kwargs["nesterov"]
		if "amsgrad" in kwargs:
			amsgrad = kwargs["amsgrad"]

		if optimizer == "sgd":
			optim = self._getSGDOptimizer(params, lr, momentum, dampening, weight_decay, nesterov)
		elif optimizer == "adadelta":
			optim = self._getAdadeltaOptimizer(params, lr, rho, eps, weight_decay)
		elif optimizer == "adagrad":
			optim = self._getAdagradOptimizer(params, lr, lr_decay, weight_decay, initial_accumulator_value)
		elif optimizer == "adam":
			optim = self._getAdamOptimizer(params, lr, betas, eps, weight_decay, amsgrad)

		return optim

	def _getAdamOptimizer(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
		_betas = (0.9, 0.999)
		_eps = 1E-08
		_weight_decay = 0.0
		_amsgrad = False

		if betas is not None:
			_betas = betas
		if eps is not None:
			_eps = eps
		if weight_decay is not None:
			_weight_decay = weight_decay
		if amsgrad is not None:
			_amsgrad = amsgrad

		return torch.optim.Adam(params, lr, betas=_betas, eps=_eps, weight_decay=_weight_decay, amsgrad=_amsgrad)

	def _getAdagradOptimizer(self, params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0):
		_lr_decay = 0.0
		_weight_decay = 0.0
		_initial_accumulator_value = 0.0

		if lr_decay is not None:
			_lr_decay = lr_decay
		if weight_decay is not None:
			_weight_decay = weight_decay
		if initial_accumulator_value is not None:
			_initial_accumulator_value = initial_accumulator_value

		return torch.optim.Adagrad(params, lr, lr_decay=_lr_decay, weight_decay=_weight_decay, initial_accumulator_value=_initial_accumulator_value)

	def _getAdadeltaOptimizer(self, params, lr, rho, eps, weight_decay):
		_rho = 0.9
		_eps = 1E-06
		_weight_decay = 0.0

		if rho is not None:
			_rho = rho
		if eps is not None:
			_eps = eps
		if weight_decay is not None:
			_weight_decay = weight_decay

		return torch.optim.Adadelta(params, lr, _rho, _eps, _weight_decay)

	def _getSGDOptimizer(self, params, lr, momentum, dampening, weight_decay, nesterov):
		_momentum = 0.0
		_dampening = 0.0
		_weight_decay = 0.0
		_nesterov = False
		if momentum is not None:
			_momentum = momentum
		if dampening is not None:
			_dampening = dampening
		if weight_decay is not None:
			_weight_decay = weight_decay
		if nesterov is not None:
			_nesterov = nesterov

		return torch.optim.SGD(params, lr, _momentum, _dampening, _weight_decay, _nesterov)




