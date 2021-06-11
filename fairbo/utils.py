import torch
from botorch.optim.initializers import initialize_q_batch
from botorch.utils.sampling import draw_sobol_samples
from matplotlib import pyplot as plt

def opt_adam_max(obj_func, bounds, N=5, nSample=500, nIter=1000, lr=0.01, display=0):

    rawX = draw_sobol_samples(bounds=bounds, n=nSample, q=1).squeeze_()
    rawY = obj_func(rawX)

    X = initialize_q_batch(rawX, rawY, N)
    X.requires_grad_(True)

    optimizer = torch.optim.Adam([X], lr)

    max_x = None
    ymax = None

    for i in range(nIter):
        optimizer.zero_grad()
        # this performs batch evaluation, so this is an N-dim tensor
        losses = -obj_func(X)  # torch.optim minimizes
        loss = losses.sum()

        loss.backward()  # perform backward pass
        optimizer.step()  # take a step

        for j, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself

        if display == 1:
            if (i + 1) % 50 == 0:
                print(f"Iteration {i+1:>3}/75 - Loss: {loss.item():>4.3f}")

        Y = obj_func(X)
        if ymax is None or torch.max(Y) > ymax:
            ymax = torch.max(Y).detach().clone()
            max_x = X[torch.argmax(Y),:].detach().clone()

    return max_x, ymax

def opt_discrete_traffic(obj_func, data, currLoc):

    currLoc = currLoc.tolist()
    nAgent = len(currLoc)

    with torch.no_grad():
        # Get all possible combinations of the next locations that can be visited by the agents.

        for i in range(nAgent):
            ne = data.get_next(currLoc[i], i, lim=4)

            if i == 0:
                index = ne.reshape(-1, 1)
            else:
                x1 = index
                x2 = ne

                index = torch.cat([x1.repeat(x2.shape[0], 1), x2.repeat(x1.shape[0], 1).t().reshape(-1, 1)], axis=1)

        length = index.shape[0]
        print("Length:", length)

        gap = 6400   # a large value may cause out-of-memory issue when calling obj_func()
        ng = int(length/gap)

        max = None
        max_x = None
        max_ind = None

        for i in range(ng+1):
            s = i*gap
            e = min((i+1)*gap, length)
            if e != s:
                X = torch.tensor(data.input_list[index[s:e, :]]).float()
                cacq = obj_func(X.reshape((e-s), -1)).detach().clone()
                curr_max = cacq.max()
                if max is None or curr_max > max:
                    argmax_index = cacq.argmax()
                    max = curr_max
                    ind = torch.arange(s, e)[argmax_index]
                    max_x = X[argmax_index].reshape(nAgent, -1).clone()
                    max_ind = index[ind]
                del X, cacq

    return max_ind, max_x


def cal_owa_regret(result, regret_rho, fmax):
    result['obsR'] =  fmax - result['obsF']
    n = result['obsF'].shape[1]
    weights = torch.pow(regret_rho, torch.arange(n))
    weights =  weights/torch.sum(weights)

    cumF =  result['obsF'].cumsum(-1)
    lambdas = cumF - result['obsF']

    fmax = (result['obsF'] + result['obsR']).mean()

    first = torch.sort(lambdas + fmax, axis=1)[0]
    first = torch.tensordot(first, weights, dims=([1],[0]))

    second = torch.sort(cumF , axis=1)[0]
    second = torch.tensordot(second, weights, dims=([1],[0]))

    ret = (first - second).cumsum(-1)

    return ret
