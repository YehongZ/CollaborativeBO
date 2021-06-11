import torch
import gpytorch
from .models import ExactGPModel
import pickle
from .testmodels import MultiLRModel, MultiCNNModel
from joblib import Parallel, delayed
from botorch.utils.transforms import normalize, unnormalize
import numpy as np
import random

class Hartmann():

    def __init__(self, dim, noise_std, seed):
        self.dim = dim
        self.noise_std = noise_std
        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])

        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]

        self.A = torch.tensor(A, dtype=torch.float)
        self.P = torch.tensor(P, dtype=torch.float)

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

    def get_func_value(self, X):
        inner_sum = torch.sum(self.A * (X.unsqueeze(1) - 0.0001 * self.P) ** 2, dim=2)
        H = torch.sum(self.alpha * torch.exp(-inner_sum), dim=1)

        return H

    def get_obs_value(self, X):
        y = self.get_func_value(X)
        return y + self.noise_std*torch.randn(y.shape, generator=self.gen)

    def get_optimal(self):
    	if self.dim == 3:
    		x = torch.tensor([0.114614, 0.555649, 0.852547])
    		y = 3.86278
    	elif self.dim == 6:
    		x = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    		y = 3.32237

    	return x, y


class RealFuncLR():

    def __init__(self, fileName, agentList):

        self.model = MultiLRModel(fileName, agentList)
        self.agentList = agentList
        self.bounds = torch.cat([torch.tensor([[20., np.log(1e-5), np.log(1e-5)]], dtype=torch.float), torch.tensor([[100., 0., 0.]])])


    def get_obs_value(self, X, s):

        if X.shape[0] != len(s):
            raise TypeError("Input no. and agent list doesn't match!")

        params = self.transfer(X)

        ret = torch.zeros(len(s))
        # for i in range(len(s)):
        #     ret[i] = torch.tensor(self.model.fit_model(params[i], s[i]))
        ret = Parallel(n_jobs=-1)(delayed(self.model.fit_model)(params[i], s[i] % 5) for i in range(len(s)))

        return torch.tensor(ret)

    def transfer(self, X):
        X = unnormalize(X, self.bounds)
        ret = X.clone().detach().cpu()
        ret[:, 0] = torch.round(ret[:, 0])  # batch_size: round to the closest integer
        ret[:, 1] = torch.exp(ret[:, 1])    # log regularization parameter
        ret[:, 2] = torch.exp(ret[:, 2])    # log learning rate

        return ret.cpu().numpy()


class RealFuncCNN():

    def __init__(self, fileName, agentList):

        self.model = MultiCNNModel(fileName, agentList)
        self.agentList = agentList
        self.bounds = torch.cat([torch.tensor([[np.log(1e-5), np.log(1e-5), np.log(1e-5)]], dtype=torch.float), torch.tensor([[0, 0., 0.]])])

    def get_obs_value(self, X, s):

        if X.shape[0] != len(s):
            raise TypeError("Input no. and agent list doesn't match!")

        params = self.transfer(X)

        # ret = torch.zeros(len(s))
        # for i in range(len(s)):
        #     ret[i] = torch.tensor(self.model.fit_model(params[i], s[i]))
        ret = Parallel(n_jobs=-1)(delayed(self.model.fit_model)(params[i], s[i] % 10) for i in range(len(s)))
        return torch.tensor(ret)

    def transfer(self, X):
        # X = unnormalize(X, self.bounds)
        ret = X.clone().detach().cpu()
        ret[:, 0] = torch.exp(ret[:, 0])  # log learning rate
        ret[:, 1] = torch.exp(ret[:, 1])    # log learning rate decay
        ret[:, 2] = torch.exp(ret[:, 2])    # log regularization parameter

        return ret.cpu().numpy()


class RealTrafficDemand():

    def __init__(self, fileName, nAgent):

        self.nAgent = nAgent

        data = pickle.load(open(fileName, "rb"))

        self.input_list = data[0][:, 0:2]
        self.output_list = data[0][:, 2]

        neg_2_mask = self.output_list == -2
        neg_1_mask = self.output_list == -1
        self.output_list = np.log1p(np.exp(self.output_list))
        self.output_list[neg_1_mask] = 0
        self.output_list[neg_2_mask] = -2

        # The optima from the traffic dataset can be obtained from the output list

        self.ne = data[1]
        self.ne[:, 1:] = self.ne[:, 1:] - 1 # index from 0

        # Each party has their own visited list. However, we will add zero demand locations visited by others too.
        self.visited_by = dict()

        self.backup = dict()
        for i in range(self.nAgent):
            self.visited_by[i] = set()
            self.backup[i] = list()

        self.freeLoc = [] # feeLoc just store all valid inputs
        self.visited = set()

        for i in range(self.ne.shape[0]):
            if self.ne[i, 0] != 0:
                self.freeLoc.append(i)

        self.nFree = len(self.freeLoc)


    def get_obs_value(self, index_list, i=None):
        # Each party has their own visited list. However, we will add zero demand locations visited by others too.

        self.visited.update(index_list.tolist())

        if i is not None:
            # initializer
            max_output = -2
            for ind in index_list:
                self.visited_by[i].add(ind.item())

                output = self.output_list[ind]
                if output > max_output:
                    max_output = output
                    ne = self.get_next(ind.item(), i, all=True)
                    self.backup[i] = ne
        else:
            for i, ind in enumerate(index_list):
                self.visited_by[i].add(ind.item())
                ne = self.get_next(ind.item(), i, all=True)
                self.backup[i] = ne

        output = torch.tensor(self.output_list[index_list], dtype=torch.float)
        mask = (output <  0).tolist()
        filt_all = index_list[mask].flatten().tolist()
        for i in range(self.nAgent):
            self.visited_by[i].update(filt_all)

        return output

    def get_rand_loc(self, n, seed=False):
       # Only called during initialization, same location can  be picked more htan once

        if seed:
            indices = torch.randint(0, self.nFree, (n,), generator=self.gen)
        else:
            indices = torch.randint(0, self.nFree, (n,))
        ret = [self.freeLoc[i] for i in indices]

        return torch.tensor(self.input_list[ret], dtype=torch.float), torch.tensor(ret, dtype=torch.int)

    # get all possible next location to visit
    def get_next(self, index, i, all=False, lim=4):
		# if all is False, return 2 * lim nearby locations at max or lim next-to-nearby locations that have not been visited by i

        def get_next_one(index):
            return  self.ne[index][1:1+self.ne[index][0]].tolist()

        def draw_random(from_list, num):
            gen = torch.Generator()
            gen.manual_seed(index)

            n = len(from_list)
            assert num < n
            ret = []
            while len(ret) < num:
                r = torch.randint(0, n, (1,), generator=gen).item()
                if from_list[r] in ret:
                    continue
                ret.append(from_list[r])
            return ret


        ne1 =  get_next_one(index)
        ne1 = [a for a in ne1 if a not in self.visited_by[i]]

        if len(ne1) > 0:
            one_step = True
            ne = ne1
        else:
            one_step = False
            ne = []
            for ind in get_next_one(index):
                ne += get_next_one(ind)
            ne = list(set(ne))
            ne2 = ne.copy()
            ne = [a for a in ne if a not in self.visited_by[i]]
            if len(ne) == 0:
                ne = []
                for ind in ne2:
                    ne += get_next_one(ind)
                ne = list(set(ne))
                ne = [a for a in ne if a not in self.visited_by[i]]

        if not all and not one_step and len(ne) > lim:
            ne = draw_random(ne, lim)
        elif not all and len(ne) > 2 * lim:
            ne = draw_random(ne, lim * 2)

        # The following almost never occur
        # If all nearby and next-to-nearby location are visited, look at what is saved in the backup or just go to any one again

        if len(ne) == 0:
            ne = self.backup[i]
            ne = [a for a in ne if a not in self.visited_by[i]]
            if not all:
                print(i, "Visited all near")
                if len(ne) > lim:
                    ne = draw_random(ne, lim)

        if len(ne) == 0:
            print("No nearby, next or backup")
            ne = self.ne[index][1:1+self.ne[index][0]].tolist()

        return torch.tensor(ne, dtype=torch.int)

    # check whether this location has been visited
    def isfree(self, index):
        if index not in self.visited:
            return True
        else:
            return False
