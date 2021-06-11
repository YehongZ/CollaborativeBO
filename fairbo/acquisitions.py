import torch
import math
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction

class UCB(AcquisitionFunction):

	def __init__(self, model, t, C1, C2):
		super(UCB2, self).__init__(model)
		self.model = model
		self.model.eval()
		self.nAgent = 1

		if not torch.is_tensor(C1):
			self.C1 = torch.tensor(C1)
		if not torch.is_tensor(C2):
			self.C2 = torch.tensor(C2)

		self.t = t

	def forward(self, X: Tensor) -> Tensor:
		posterior = self.model(X)
		mean = posterior.mean
		variance = posterior.variance

		d = self.model.train_inputs[0].shape[-1]
		beta = self.C1*d*torch.log(torch.tensor(self.C2*self.t))

		delta = (beta * variance).sqrt()

		return mean + delta


class FairBatchOWAUCB(AcquisitionFunction):

	def __init__(self, model, nAgent, t, ws, lambd, C1, C2, sort_weights=True):
		super(FairBatchOWAUCB, self).__init__(model)
		self.model = model
		self.model.eval()
		self.lambd = lambd
		self.nAgent = nAgent
		self.t = t
		self.ws = ws # weights; decreasing
		if sort_weights:
			self.ws = torch.sort(self.ws, descending=True)[0]

		self.C1 = torch.tensor(C1)
		self.C2 = torch.tensor(C2)

	def forward(self, X: Tensor) -> Tensor:

		if not X.shape[-1] == self.nAgent*self.model.train_inputs[0].shape[-1]:
			raise TypeError('nAgent: Input dimension mismatch')

		combined = X.reshape(X.shape[0] , self.nAgent, -1)
		posterior = self.model(combined)
		joint_mean = posterior.mean
		joint_cov = posterior.covariance_matrix

		item1 = torch.sort(joint_mean, dim=1, descending=True)[0] +  torch.sort(self.lambd)[0]
		item1 = torch.sort(item1, dim=1)[0] * self.ws
		item1 = torch.sum(item1, dim=1)

		noise = self.model.likelihood.noise
		n = torch.sum(self.ws**2) # Cauchy Schwarz
		d = self.model.train_inputs[0].shape[-1]
		alpha = self.C1*n*d*torch.log(torch.tensor(self.C2*self.t))
		item2 = 0.5*torch.logdet(torch.eye(self.nAgent)+joint_cov*(1./noise))
		item2 = torch.sqrt(alpha*item2)

		return item1 + item2

class FairBatchOWAUCB_NoPermute(AcquisitionFunction):

	def __init__(self, model, nAgent, t, ws, lambd, C1, C2):
		super(FairBatchOWAUCB_NoPermute, self).__init__(model)
		self.model = model
		self.model.eval()
		self.lambd = lambd
		self.nAgent = nAgent
		self.t = t
		self.ws = ws # weights; decreasing

		self.C1 = torch.tensor(C1)
		self.C2 = torch.tensor(C2)

	def forward(self, X: Tensor) -> Tensor:

		if not X.shape[-1] == self.nAgent*self.model.train_inputs[0].shape[-1]:
			raise TypeError('nAgent: Input dimension mismatch')

		n = torch.sum(self.ws**2) # Cauchy Schwarz
		d = self.model.train_inputs[0].shape[-1]
		alpha = self.C1*n*d*torch.log(self.C2*self.t)

		combined = X.reshape(X.shape[0] , self.nAgent, -1)
		posterior = self.model(combined)
		joint_mean = posterior.mean
		joint_cov = posterior.covariance_matrix

		item1 = joint_mean + self.lambd
		item1 = torch.sort(item1, dim=1)[0] * self.ws
		item1 = torch.sum(item1, dim=1)

		noise = self.model.likelihood.noise
		item2 = 0.5*torch.logdet(torch.eye(self.nAgent)+joint_cov*(1./noise))
		item2 = torch.sqrt(alpha*item2)

		return item1+item2
