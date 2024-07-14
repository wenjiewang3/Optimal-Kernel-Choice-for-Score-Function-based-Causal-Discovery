import numpy as np
import torch
import torch.nn as nn

class Marg_model(nn.Module):
    def __init__(self, PA, X, param=None, noise_scale = np.sqrt(0.1)):
        super().__init__()
        self.param = param
        self.device = param["device"]
        self.Xinit = param.get('width_init')

        self.n = PA.shape[0]
        self.pa_list = param['pa_list']
        # kernel init
        PAkernel_width = self.MultiKernelInit_multi(PA, self.pa_list)
        Xkernel_width = self.Onekernel_median(X, self.Xinit)


        self.PAlength_scale_ = nn.Parameter(torch.tensor(PAkernel_width).to(self.device))
        self.PAlength_scale_.requires_grad = True

        self.amp = 1

        self.noise_scale_ = nn.Parameter(torch.tensor(noise_scale).to(self.device))
        self.noise_scale_.requires_grad = True

        self.Xlength_scale_ = torch.tensor(Xkernel_width).to(self.device)
        self.score = 1e5

    def MultiKernelInit_multi(self, PA, pa_list):
        if PA.max() == PA.min():
            return [10.]
        n = len(pa_list)
        width_list = np.random.randn(n)
        for i in range(n):
            PA_sub = PA[:, pa_list[i]]
            PA_sub = np.mat(PA_sub.numpy())
            T = PA_sub.shape[0]
            GX = np.sum(np.multiply(PA_sub, PA_sub), axis=1)
            Q = np.tile(GX, (1, T))
            R = np.tile(GX.T, (T, 1))
            dists = Q + R - 2 * PA_sub * PA_sub.T
            dists = dists - np.tril(dists)
            dists = np.reshape(dists, (T**2, 1))
            width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
            width = width * 2.5
            width_list[i] = width
        return width_list

    def Onekernel_median(self, X, weight=None):
        if weight is not None:
            return torch.tensor(weight).numpy()
        X = np.mat(X.numpy())
        T = X.shape[0]
        GX = np.sum(np.multiply(X, X), axis=1)
        Q = np.tile(GX, (1, T))
        R = np.tile(GX.T, (T, 1))
        dists = Q + R - 2 * X * X.T
        dists = dists - np.tril(dists)
        dists = np.reshape(dists, (T**2, 1))
        width = np.sqrt(0.5 * np.median(dists[np.where(dists > 0)], axis=1)[0, 0])
        width = width * 2.5  # kernel width
        return width

    def predict(self, PA):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        alpha = self.alpha
        k = self.PAkernel_multi(PA, self.pa_list)
        mu = k.T.mm(alpha)
        return mu

    def fit(self, PA, X):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        D = PA.shape[1]
        Kpa = self.cal_PAkernel(PA)
        KX = self.Xkernel(X)

        L = torch.linalg.cholesky(Kpa)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, KX))

        marginal_likelihood = (-0.5 * torch.trace(KX.mm(alpha))
                               - D*torch.log(torch.diag(L)).sum()
                               - D*D * 0.5 * np.log(2 * np.pi))

        self.PA = PA
        self.X = X
        self.L = L
        self.alpha = alpha
        self.Kpa = Kpa
        self.KX = KX
        Score = -marginal_likelihood.sum()
        self.score = Score.detach().item()
        return Score

    def cal_PAkernel(self, PA):

        if self.PAlength_scale_.requires_grad:
            length_scale = self.PAlength_scale_.data
            length_scale = length_scale.clamp_(0.5, 5)
            self.PAlength_scale_.data = length_scale

        noise_scale = self.noise_scale_.data
        noise_scale = noise_scale.clamp_(1e-2, 1)
        self.noise_scale_.data = noise_scale


        Kpa = self.PAkernel_multi(PA, self.pa_list)
        K = Kpa + self.noise_scale_ * torch.eye(PA.shape[0], device=self.device)
        K[K<0] = 0
        return K

    def Xkernel(self, X):
        X = X / self.Xlength_scale_
        Xsq = (X ** 2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Xsq.T - 2 * X.mm(X.T)
        Kx =  torch.exp(- 0.5 * sqdist)
        return Kx

    def PAkernel_multi(self, PA, pa_list):
        n = len(pa_list)
        for i in range(n):
            sub_PA = PA[:, pa_list[i]]
            sPA_scaled = sub_PA/self.PAlength_scale_[i]
            sPAsq = (sPA_scaled ** 2).sum(dim=1, keepdim=True)
            temp_dist =  sPAsq + sPAsq.T - 2 * sPA_scaled.mm(sPA_scaled.T)
            if i == 0:
                sqdist = temp_dist
            else:
                sqdist += temp_dist
        Kpa = torch.exp(-0.5*sqdist)
        return Kpa


    def train_step(self, PA, x, opt):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        # range
        PA = PA.to(self.device)
        x = x.to(self.device)
        opt.zero_grad()
        nll = -self.fit(PA, x)
        nll.backward()
        opt.step()
        return {
            "score": nll.item(),
        }


