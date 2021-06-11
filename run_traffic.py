from fairbo.functions import RealTrafficDemand
from fairbo.utils import opt_discrete_traffic
import torch
from fairbo.models import ExactGPModel
from fairbo.acquisitions import FairBatchOWAUCB_NoPermute
import gpytorch
import numpy as np
import datetime as dt
import time
import gc

from argparse import ArgumentParser

def main():
    argparser = ArgumentParser()
    argparser.add_argument('--c1', type=float)
    argparser.add_argument('--nAgent', type=int)
    argparser.add_argument('--vary', dest='vary', default=False, action='store_true', help='vary c1')
    args = argparser.parse_args()

    C1 = args.c1
    if C1 is None:
        C1 = 0.1
    C2 = 1.
    vary = args.vary

    nAgent = args.nAgent
    if nAgent is None:
        nAgent = 8

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path ='./results/'

    d = 2
    ns = 6

    nT = 60
    nLoop = 10

    seed = [911, 71, 44, 2525, 6023, 556, 3399, 4863, 5269, 9973]

    for base in [1,0.8,0.6,0.4,0.2]:
        ws = torch.pow(base, torch.arange(nAgent))  # geometric weights

        experimentNO = 'traffic_ns={}_nAgent={}_base={}_C1={}_vary={}'.format(ns, nAgent, base, C1, vary)

        with torch.no_grad():
            obsXAll = torch.empty(nLoop, nAgent, nT, d)
            obsYAll = torch.empty(nLoop, nAgent, nT)

        for l in range(nLoop):

            torch.manual_seed(seed[l])
            func = RealTrafficDemand("./data/tlog44.pkl", nAgent)

            get_obs = func.get_obs_value

            with torch.no_grad():
                obsX = torch.empty(nAgent, ns, d)
                obsY = torch.empty(nAgent, ns)
                obsInd = torch.empty(nAgent, ns, dtype=int)

                for i in range(nAgent):
                    while True:
                        obsX[i], index = func.get_rand_loc(ns)
                        obsY[i] = get_obs(index, i)
                        obsInd[i] = index
                        if torch.any(obsY[i] > 0):
                            break

                trainX = obsX.reshape(-1, d)
                trainY = obsY.reshape(-1)

            boModel = ExactGPModel(trainX, trainY)
            hypers = {
                'mean_module.constant': 1.4646,
                'likelihood.noise_covar.noise': 0.0117,
                'covar_module.outputscale': 0.7969,
                'covar_module.base_kernel.lengthscale': [0.6276, 0.6490]
            }

            boModel.initialize(**hypers)
            boModel.set_train_data(trainX, trainY)
            #boModel.train_model()

            start_time = dt.datetime.now()

            for t in range(ns, nT):

                lambd = torch.sum(obsY, 1)

                if vary:
                    C1 = C1 * ratio
                acq_func = FairBatchOWAUCB_NoPermute(boModel, nAgent, t+1, ws, lambd, C1, C2)

                if t == ns:
					# Initialize to best location, whose demand (not log demand) > 1
                    currLoc = obsInd[torch.arange(nAgent),torch.argmax(obsY, dim=1)]
                else:
                    currLoc = obsInd[:, -1]

                newInd, newX = opt_discrete_traffic(acq_func, func, currLoc)

                newY = get_obs(newInd)

                p = 10
                if t%p == 0:
                    end_time = dt.datetime.now()
                    elapsed_time = (end_time - start_time).seconds/p

                    print('Loop', l+1, ': ', t+ns, ' observations selected.', '  Time per iter: ', elapsed_time, 's')
                    start_time = dt.datetime.now()

                obsX = torch.cat([obsX, newX.unsqueeze(1)], 1)
                obsY = torch.cat([obsY, newY.unsqueeze(1)], 1)
                obsInd = torch.cat([obsInd, newInd.unsqueeze(1)], 1)


                posterior_mean = boModel(newX).mean
                boModel = boModel.get_fantasy_model(newX, newY)

                del acq_func, lambd, newY, newX, newInd,
                gc.collect()


            obsXAll[l] = obsX
            obsYAll[l] = obsY

            result = {}
            result['obsX'] = obsXAll
            result['obsY'] = obsYAll

            torch.save(result, path+'result_'+experimentNO+'.pt')

if __name__ == '__main__':
    main()
