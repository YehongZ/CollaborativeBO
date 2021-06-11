import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fairbo.functions import RealFuncLR, RealFuncCNN
from fairbo.utils import opt_adam_max
import torch
from fairbo.models import ExactGPModel
from botorch.utils.sampling import draw_sobol_samples
from fairbo.acquisitions import  FairBatchOWAUCB,  FairBatchOWAUCB_NoPermute
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import time

from argparse import ArgumentParser

def main():
    argparser = ArgumentParser()
    argparser.add_argument('--exp', type=str, required=True, choices=['lr', 'cnn'])
    argparser.add_argument('--c1', type=float, required=True)
    argparser.add_argument('--c2', type=float, required=True)
    argparser.add_argument('--n', type=int, required=True)
    argparser.add_argument('--vary', dest='vary', default=False, action='store_true', help='vary c1')
    args = argparser.parse_args()

    exp = args.exp
    nAgent = args.n
    C1 = args.c1
    C2 = args.c2
    vary = args.vary
    agentList = list(range(nAgent))

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d = 3

    if exp == 'lr':
        # Linear Regression model with mobile sensor dataset
        # input: three hyperparameters
        #        batch size [20, 100]
        #        regularization parameter log[1e-5, 1]
        #        learning rate log[1e-5, 1]
        bounds = torch.cat([torch.zeros(1, d), torch.ones(1, d)])
        # the bound would be normalized in RealFuncLR
        func = RealFuncLR("./data/har_data_mixed.pkl", agentList)
        hypers = {
            'likelihood.noise_covar.noise': 0.01,
            'covar_module.outputscale': 0.13,
            'covar_module.base_kernel.lengthscale': [1.6, 0.44, 0.26],
            'mean_module.constant': 0.5,
        }
    elif exp == 'cnn':
        # CNN model with emnist dataset
        # input: three hyperparameters
        #        learning rate log[1e-5, 1]
        #        learning rate decay log[1e-5, 1]
        #        regularization parameter log[1e-5, 1]
        bounds = torch.cat([torch.tensor([[np.log(1e-5), np.log(1e-5), np.log(1e-5)]], dtype=torch.float), torch.tensor([[0, 0., 0.]])])
        func = RealFuncCNN("./data/emnist_data_mixed.pkl", agentList)
        hypers = {
            'likelihood.noise_covar.noise': 0.01,
            'covar_module.outputscale': torch.tensor(0.12),
            'covar_module.base_kernel.lengthscale': [1.6, 3.2, 3.6],
            'mean_module.constant': 0.3,
        }


    get_obs = func.get_obs_value
    nAgent = len(agentList)


    ns = 2
    nT = 50
    nLoop = 10

    seed = [8911, 7, 444, 3525, 5023, 1556, 5399, 7863, 4269, 2973]
    C2 = 1.
    path ='./results/'

    for base in [1,0.8,0.6,0.4,0.2]:
        ws = torch.pow(base, torch.arange(nAgent))

        experimentNO = exp + '_base={}_c1={}_vary={}'.format(base, C1, vary)

        with torch.no_grad():
            obsXAll = torch.empty(nLoop, nAgent, nT, d)
            obsYAll = torch.empty(nLoop, nAgent, nT)

        for l in range(nLoop):

            torch.manual_seed(seed[l])

            with torch.no_grad():
                obsX = torch.empty(nAgent, ns, d)
                obsY = torch.empty(nAgent, ns)
                obsF = torch.empty(nAgent, ns)

                for i in range(nAgent):
                    obsX[i] = draw_sobol_samples(bounds=bounds, n=ns, q=1).squeeze_()
                    obsY[i] = get_obs(obsX[i], [agentList[i]]*ns)#, agentList[i])

                trainX = obsX.reshape(-1, d)
                trainY = obsY.reshape(-1)

            boModel = ExactGPModel(trainX, trainY)
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(0.1).pow(2),
                'covar_module.outputscale': trainY.mean().clone().detach().requires_grad_(True),
            }

            boModel.initialize(**hypers)
            boModel.set_train_data(trainX, trainY)
            # boModel.train_model()

            start_time = dt.datetime.now()

            for t in range(ns, nT):
                sorton = torch.sum(obsY, 1)
                lambd = torch.sum(obsY, 1)

                ratio = torch.sum(ws)**2 / (torch.sum(ws**2) * nAgent)
                if vary:
                    C1 = C1 * ratio
                if base < 1:
                    acq_func = FairBatchOWAUCB(boModel, nAgent, t+1, ws, lambd, C1=C1 , C2=C2)
                else:
                    acq_func = FairBatchOWAUCB_NoPermute(boModel, nAgent, t+1, ws, lambd, C1=C1, C2=C2)

                nbounds = torch.cat(nAgent*[bounds], 1)
                ret = opt_adam_max(acq_func, nbounds, nIter=500, N=20, nSample=1500)
                del acq_func


                with torch.no_grad():

                    newX = ret[0].reshape(nAgent, -1).detach()

                    if base < 1:

                        rank = torch.argsort(torch.argsort(sorton))

                        posterior_mean = boModel(newX).mean
                        posterior_argsort = torch.argsort(posterior_mean, descending=True)

                        newX = newX[posterior_argsort]
                        newX = newX[rank]

                    newY = get_obs(newX, agentList)

                    if t < 5:
                        p = 1
                    else:
                        p = 10

                    if t%p == 0:
                        end_time = dt.datetime.now()
                        elapsed_time= (end_time - start_time).seconds/p

                        print('Loop', l+1, ': ', t, ' observations selected.', '  Time per iter: ', elapsed_time, 's')
                        start_time = dt.datetime.now()

                    obsX = torch.cat([obsX, newX.unsqueeze(1)], 1)
                    obsY = torch.cat([obsY, newY.unsqueeze(1)], 1)

                    boModel = boModel.get_fantasy_model(newX, newY)


            obsXAll[l] = obsX
            obsYAll[l] = obsY

            result = {}
            result['obsX'] = obsXAll
            result['obsY'] = obsYAll

            torch.save(result, path + 'result_'+ experimentNO+'.pt')

if __name__ == '__main__':
    main()
