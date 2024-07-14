from typing import List
import numpy as np
import torch
from numpy import ndarray
from torch.optim import Adam
from Model.MIModel import MIModel
from Utils.lbfgsb_scipy import LBFGSBScipy
import warnings
warnings.filterwarnings("ignore")

def local_score_mi_lbfgsb(Data: ndarray, Xi: int, PAi: List[int], param = None):
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


    local_score_model = MIModel(PA, X, param).to(param['device'])

    # Adam
    if param['optim'] == 'adam':
        optim = Adam(local_score_model.parameters(), lr=0.005)
        # from torch.optim import SGD
        # optim = SGD(general_score_model.parameters(), lr=1e-5)
        for i in range(param['epochs']):
            d_train = local_score_model.train_step(PA, X, optim)

    else:
    # lbfgsb
        optim = LBFGSBScipy(local_score_model.parameters())
        def closure():
            optim.zero_grad()
            Score, nlml, Jac = local_score_model.Score(PA, X)
            Score.backward()
            return Score
        optim.step(closure)

    return local_score_model.score
