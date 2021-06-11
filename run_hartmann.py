from fairbo.functions import Hartmann
from fairbo.utils import opt_adam_max
import torch
from fairbo.models import ExactGPModel
from botorch.utils.sampling import draw_sobol_samples
from fairbo.acquisitions import  FairBatchOWAUCB,  FairBatchOWAUCB_NoPermute
import gpytorch
from matplotlib import pyplot as plt
import datetime as dt
import time

from argparse import ArgumentParser

def main():
    argparser = ArgumentParser()
    argparser.add_argument('--c1', type=float)
    argparser.add_argument('--c2', type=float)
    argparser.add_argument('--noise', type=float)
    argparser.add_argument('--vary', dest='vary', default=False, action='store_true', help='vary c1')
    args = argparser.parse_args()

    C1 = args.c1
    if C1 is None:
        C1 = 0.08
    C2 = args.c2
    if C2 is None:
        C2 = 5.
    noise_std = args.noise
    if noise_std is None:
        noise_std = 0.1
    vary = args.vary

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = './results/'
    nAgent = 3 #fix

    ns = 10    # no. of initial points for each agent
    nT = 70    # no. of BO iterations
    nLoop = 10
    seeds = [8911, 7, 444, 3525, 5023, 1556, 5399, 7863, 4269, 2973]
    d = 6
    bounds = torch.cat([torch.zeros(1, d), torch.ones(1, d)])

    for base in [1,0.8,0.6,0.4,0.2]:
        ws = torch.pow(base, torch.arange(nAgent))

        experimentNO = 'hartmann_nAgent={}_OWA_base={}_C1={}_std={}'.format(nAgent, base, C1, noise_std)

        with torch.no_grad():
            obsXAll = torch.empty(nLoop, nAgent, nT, d)
            obsYAll = torch.empty(nLoop, nAgent, nT)
            obsFAll = torch.empty(nLoop, nAgent, nT)
            obsLambdAll = torch.empty(nLoop, nAgent, nT)

        for loop in range(nLoop):
            seed = seeds[loop]
            syn = Hartmann(d, noise_std, seed)
            get_obs = syn.get_obs_value
            get_func = syn.get_func_value
            torch.manual_seed(seed)

            with torch.no_grad():
                obsX = torch.empty(nAgent, ns, d)
                obsY = torch.empty(nAgent, ns)
                obsF = torch.empty(nAgent, ns)

                initialX = draw_sobol_samples(bounds=bounds, n=ns*nAgent, q=1).squeeze_()
                initialF = get_obs(initialX)

                for i in range(nAgent):
                    obsX[i] = initialX[i*ns:i*ns+ns]
                    obsY[i] = get_obs(obsX[i])    # noisy observations
                    obsF[i] = get_func(obsX[i])    # function values without noise

                trainX = obsX.reshape(-1, d)
                trainY = obsY.reshape(-1)


            boModel = ExactGPModel(trainX, trainY)
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(noise_std).pow(2),
                'covar_module.outputscale': torch.tensor(0.588),
                'covar_module.base_kernel.lengthscale': [0.3, 0.3967, 0.9883, 0.3209, 0.3867, 0.3243],
                'mean_module.constant':  0.157
            }
            boModel.initialize(**hypers)
            boModel.set_train_data(trainX, trainY)
            # boModel.train_model(display=1)


            # Record the results
            fmaxR = obsF.max(1).values.unsqueeze(1)    # max(f(x_i)) for each agent

            start_time = dt.datetime.now()
            for t in range(ns, nT):

                with torch.no_grad():
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
                ret = opt_adam_max(acq_func, nbounds, nIter=150, N=20, nSample=1500, lr=0.02)
                del acq_func

                with torch.no_grad():
                    newX = ret[0].reshape(nAgent, -1).detach()

                    if base < 1:
                        rank = torch.argsort(torch.argsort(sorton))

                        posterior_mean = boModel(newX).mean
                        posterior_argsort = torch.argsort(posterior_mean, descending=True)

                        newX = newX[posterior_argsort]
                        newX = newX[rank]

                    newY = get_obs(newX)
                    newF = get_func(newX)

                    obsX = torch.cat([obsX, newX.unsqueeze(1)], 1)
                    obsY = torch.cat([obsY, newY.unsqueeze(1)], 1)
                    obsF = torch.cat([obsF, newF.unsqueeze(1)], 1)

                    boModel = boModel.get_fantasy_model(newX, newY)

                    fmaxR = torch.cat([fmaxR, obsF.max(1).values.unsqueeze(1)], 1)

                    p = 10
                    if t%p == 0:

                        end_time = dt.datetime.now()
                        elapsed_time= (end_time - start_time).seconds/p

                        print('Loop', loop+1, ', ', t, ' observations selected.', '  Time per iter: ', elapsed_time, 's')
                        print("c1", C1, 'Current maximal f values: ', str(fmaxR[:, -1].cpu().numpy()))
                        start_time = dt.datetime.now()


            obsXAll[loop] = obsX
            obsYAll[loop] = obsY
            obsFAll[loop] = obsF

        result = {}
        result['obsF'] = obsFAll
        result['obsX'] = obsXAll
        result['obsY'] = obsYAll
        torch.save(result, path+'result_'+experimentNO+'.pt')

if __name__ == '__main__':
    main()
