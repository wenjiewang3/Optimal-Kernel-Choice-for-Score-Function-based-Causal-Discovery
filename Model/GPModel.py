import numpy as np
import torch
import torch.nn as nn


class GPmodel(nn.Module):
    def __init__(self, PA):
        super().__init__()
        self.amp = 1

        self.length_scale_ = nn.Parameter(torch.tensor(self.PAkernel_init(PA)))
        self.length_scale_.requires_grad = True

        self.noise_scale_ = nn.Parameter(torch.tensor(0.1))
        self.noise_scale_.requires_grad = True

        self.score = 1e5

    def PAkernel_init(self, X, weight=None):
        if weight is not None:
            return torch.tensor(weight).numpy()
        sq = (X ** 2).sum(dim=1, keepdim=True)
        sqdist = sq + sq.T - 2 *X.mm(X.T)
        if (sqdist.max() == 0):
            return 10.
        dists = (sqdist - torch.tril(sqdist, diagonal=0)).flatten()
        mid = torch.median(dists[dists>0])
        width = mid  # kernel width
        return width.detach().numpy()

    def predict(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        alpha = self.alpha
        k = self.kernel_mat(self.X, x)
        mu = k.T.mm(alpha)
        return mu

    def fit(self, X, y):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        X = X.float()
        y = y.float()

        N = X.shape[0]
        K = self.kernel_mat_self(X)
        L = torch.linalg.cholesky(K)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        marginal_likelihood = (
            -0.5 * torch.trace(y.T.mm(alpha))
            - N*torch.log(torch.diag(L)).sum()
            - N*N * 0.5 * np.log(2 * np.pi)
        )

        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha
        self.K = K
        self.score = -marginal_likelihood.cpu().detach().numpy()
        return marginal_likelihood

    def kernel_mat_self(self, X):
        if self.length_scale_.requires_grad:
            length_scale = self.length_scale_.data
            length_scale = length_scale.clamp_(0.5, 5)
            self.length_scale_.data = length_scale

        noise_scale = self.noise_scale_.data
        noise_scale = noise_scale.clamp_(1e-2, 1)
        self.noise_scale_.data = noise_scale

        self.Kpa = self.PAkernel(X)
        K = self.Kpa + self.noise_scale_ * torch.eye(len(X))
        K[K<0] = 0
        return K

    def PAkernel(self, PA):
        PA = PA / self.length_scale_
        PAsq = (PA ** 2).sum(dim=1, keepdim=True)
        sqdist = PAsq + PAsq.T - 2 * PA.mm(PA.T)
        Kpa =  torch.exp(- 0.5 * sqdist)
        return Kpa

    def kernel_mat(self, X, Z):
        Xsq = (X**2).sum(dim=1, keepdim=True)
        Zsq = (Z**2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        return self.amp * torch.exp(-0.5 * sqdist / self.length_scale)

    def train_step(self, X, y, opt):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        # range
        opt.zero_grad()
        nll = -self.fit(X, y).sum()
        nll.backward()
        opt.step()
        return {
            "score": nll.item(),
        }