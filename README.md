# CollaborativeBO
This is the code for the following paper:

"Collaborative Bayesian Optimization with Fair Regret". Rachael Hwee Ling Sim, Yehong Zhang, Bryan Kian Hsiang Low and Patrick Jaillet. In Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.

## Requirements

python >= 3.7

pytorch >= 1.6.0

gpytorch

botorch

matplotlib

tenorflow 2

pickle

## Experiments

Hartmann-6d function
```
python run_hartmann.py --c1 0.08 --c2 5 --noise 0.01
```

Hyperparameter tuning of LR with mobile sensor dataset
```
python run_hyp_tuning.py  --c1 0.01 --c2 10 --exp lr
```

Hyperparameter tuning of CNN with FEMNIST
```
python run_hyp_tuning.py  --c1 0.001 --c2 1 --exp cnn
```

Mobility demand hotspot discovery on traffic dataset.
```
python run_traffic.py  --c1 0.1 --c2 1 --nAgent 8
```

The implementation of our Fair BO algorithm is mainly inside the fairbo folder.
- `acquisitions.py` contains the acquisition function
- `functions.py` contains the objective functions for querying
- `models.py` contains the underlying GP model used in BO.  
- `testmodels.py` contains the models of LR and CNN for hyperparameters tuning.
- `utils.py` contains functions for the optimization step and analysing the results e.g. cumulative regret, variance

Other experiment settings e.g. the choice of weights, c1, c2, number of initial observations and time horizon are set within `run_x.py`