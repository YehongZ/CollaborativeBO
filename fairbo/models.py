import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        super(ExactGPModel, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_pred_mean(self, x):
    	return self.__call__(x).mean

    def get_pred_variance(self, x):
    	return self.__call__(x).variance

    def train_model(self, nIter=500, lr=0.01, display=0):

        likelihood = self.likelihood
        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(nIter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_inputs[0])
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_targets)
            loss.backward()

            if display == 1:
                if i%50 == 0:
                    print('Iter %d/%d - Loss: %.3f  noise: %.4f  signal variance: %.3f' % (
                        i + 1, nIter, loss.item(),
                        self.likelihood.noise.item(),
                        self.covar_module.outputscale.item(),
                    ))

            optimizer.step()
