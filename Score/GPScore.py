from typing import List
import numpy as np
import torch
from numpy import ndarray
from torch.optim import Adam
import warnings

from Model.GPModel import GPmodel
from Utils.lbfgsb_scipy import LBFGSBScipy

warnings.filterwarnings("ignore")

def local_score_gp_lbfgsb(Data: ndarray, Xi: int, PAi: List[int], param = None):
    var_idx = param['var_idx']
    X = torch.tensor(Data[:, var_idx[Xi]])
    if len(PAi):
        pa_ids_list = []
        data_idx = []
        idx = 0
        for i in PAi:
            PA_temp = var_idx[i]
            data_idx += PA_temp
            # current idx
            PA_temp_dim = len(PA_temp)
            temp_list = []
            for tt in range(idx, idx+PA_temp_dim):
                temp_list.append(tt)
            idx += PA_temp_dim
            pa_ids_list.append(temp_list)

        param['pa_list'] = pa_ids_list
        PA = torch.tensor(Data[:, data_idx])
    else:
        PA = torch.tensor(np.zeros((Data.shape[0], 1)))
        param['pa_list'] = [[0]]


    local_score_model = GPmodel(PA)

    # Adam
    if param['optim'] == 'adam':
        param['epochs'] = 300
        optim = Adam(local_score_model.parameters(), lr=0.05)
        for i in range(param['epochs']):
            d_train = local_score_model.train_step(PA, X, optim)
    else:
        # lbfgsb
        optim = LBFGSBScipy(local_score_model.parameters())
        def closure():
            optim.zero_grad()
            nlml = -local_score_model.fit(PA, X)
            nlml.backward()
            return nlml
        optim.step(closure)

    return local_score_model.score
