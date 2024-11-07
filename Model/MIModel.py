import numpy as np
import torch
import torch.nn as nn

class MIModel(nn.Module):
    def __init__(self, PA, X, param, noise_scale = np.sqrt(0.1)):
        super().__init__()
        self.param = param
        self.device = param['device']
        self.Xinit = param.get('width_init')
        self.threshold = param['threshold']

        self.n = PA.shape[0]
        self.pa_list = param['pa_list']

        PAkernel_width = self.MultiKernelInit_multi(PA, self.pa_list)
        Xkernel_width = self.Onekernel_median(X, self.Xinit)

        self.PAlength_scale_ = nn.Parameter(torch.tensor(PAkernel_width).to(self.device))
        self.PAlength_scale_.requires_grad = True

        self.noise_scale_ = nn.Parameter(torch.tensor(noise_scale).to(self.device))
        self.noise_scale_.requires_grad = True

        self.Xlength_scale_ = nn.Parameter(torch.tensor(Xkernel_width).to(self.device))
        self.Xlength_scale_.requires_grad = True
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
            width_list[i] = width#/len(pa_list[i])
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

    def cal_PAkernel(self, PA):
        if self.PAlength_scale_.requires_grad:
            length_scale = self.PAlength_scale_.data
            length_scale = length_scale.clamp_(0.2, 5)
            self.PAlength_scale_.data = length_scale

        if self.Xlength_scale_.requires_grad:
            # print("Xgrad: ", self.Xlength_scale_.grad)
            xlength_scale = self.Xlength_scale_.data
            xlength_scale = xlength_scale.clamp_(0.2, 5)
            self.Xlength_scale_.data = xlength_scale

        if self.noise_scale_.requires_grad:
            noise_scale = self.noise_scale_.data
            noise_scale = noise_scale.clamp_(1e-2, 10)
            self.noise_scale_.data = noise_scale

        Kpa = self.PAkernel_multi(PA, self.pa_list)
        K =  Kpa + self.noise_scale_ * torch.eye(PA.shape[0], device=self.device)
        K[K<0] = 0
        return K
#
    def Xkernel(self, X):
        Xsq = (X ** 2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Xsq.T - 2 * X.mm(X.T)
        Kx =  torch.exp(- 0.5 * sqdist / self.Xlength_scale_**2)
        return Kx

    def PAkernel_multi(self, PA, pa_list):
        n = len(pa_list)
        for i in range(n):
            PAt = PA[:, pa_list[i]]
            PAt = PAt/self.PAlength_scale_[i]
            PAtsq = (PAt ** 2).sum(dim=1, keepdim=True)
            temp_dist = PAtsq + PAtsq.T - 2 * PAt.mm(PAt.T)
            if i == 0:
                sqdist = temp_dist
            else:
                sqdist += temp_dist

        Kpa = torch.exp(-0.5*sqdist)
        return Kpa

    def ConditionalLikelihood(self, PA, X):
        n = X.shape[0]
        Kpa = self.cal_PAkernel(PA)
        KX = self.Xkernel(X)

        L = torch.linalg.cholesky(Kpa)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, KX))
        marginal_likelihood = (-0.5*torch.trace(KX.mm(alpha))
                               - n*(torch.log(torch.diag(L)).sum() + n * 0.5 * np.log(2 * np.pi)))
        self.Kpa = Kpa; self.KX = KX
        self.PA = PA; self.X = X
        self.L = L; self.alpha = alpha
        return marginal_likelihood / n

    def Jacobian(self, X):
        """
        determinent of Jabobian matrix term
        param y: test input data point. N x 1.
        """

        n = X.shape[0]
        Kx = self.KX
        xm = X.T.unsqueeze(2)
        Xm = X.T.unsqueeze(1)
        dist = (xm - Xm)[0]

        driv_Kx = - torch.mul(dist, Kx) / self.Xlength_scale_**2
        abs_deriv_diag = torch.abs(driv_Kx)
        nonZerosIdx = abs_deriv_diag > self.threshold
        log_deriv_diag = torch.log(abs_deriv_diag[nonZerosIdx])
        driv_score = torch.sum(log_deriv_diag)
        return driv_score / n

    def multi_Jacobian(self, X):
        n = X.shape[0]
        v = X.shape[-1]
        Kx = self.KX

        xm = X.T.unsqueeze(2)
        Xm = X.T.unsqueeze(1)
        dist2 = (xm - Xm)
        dist = dist2.permute(1, 2, 0)

        dist = -dist/(self.Xlength_scale_**2)
        Kx = torch.unsqueeze(Kx, dim=2)
        driv_mat = torch.mul(dist, Kx)
        driv_mat = torch.sum(driv_mat**2, dim=2)
        nonZerosIdx = driv_mat > self.threshold
        driv_mat = torch.sqrt(driv_mat[nonZerosIdx])
        log_deriv_diag = torch.log(driv_mat)

        # nonZerosIdx = driv_mat > self.threshold
        # driv_mat = driv_mat[nonZerosIdx]
        weight = torch.tensor(self.n**2)/torch.tensor(driv_mat.size())
        driv_score = torch.sum(log_deriv_diag)*weight
        return driv_score / n


    def Score(self, x, y):
        """
        score function using marginal likelihood of joint distribution
        :param x: independent variable. N x m
        :param y: dependent variable. N x 1
        :return:
        """
        n = x.shape[0]
        x = x.to(self.device)
        y = y.to(self.device)
        marginal_likelihood = self.ConditionalLikelihood(x, y)/n
        # driv_term1 = self.Jacobian(y)/n
        driv_term2 = self.multi_Jacobian(y)/n
        driv_term = driv_term2
        nll = -(marginal_likelihood + driv_term)
        self.score = nll.detach().numpy()
        return nll, marginal_likelihood, driv_term

    def train_step(self, x, y, opt):
        opt.zero_grad()
        score, nlml, Jacobian = self.Score(x, y)
        score.backward()
        # print("Xgrad:", self.Xlength_scale_.grad)
        opt.step()
        return {'score': score.item(),
                'nlml': nlml.detach().cpu(),
                'driv': Jacobian.detach().cpu(),
                'PAlength': self.PAlength_scale_.data,
                'Xlength': self.Xlength_scale_.data,
                }




