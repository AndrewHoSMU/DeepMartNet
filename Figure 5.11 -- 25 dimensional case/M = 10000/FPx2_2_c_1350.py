#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
print(device)

# number of dimensions
Dim = 25

# number of grids over time
if device == "cpu":
    N = 1350
else:
    N = 1350

# simulation time
T = 9.

# number of trajectories
if device == "cpu":
    if Dim == 1:
        M = 250
    else:
        M = 10000
else:
    if Dim == 1:
        M = 500
    else:
        M = 10000


def sample_NdFP(M, T, N, Dim):
    dt = T / N
    Gaussvar = np.sqrt(dt)
    arr = torch.randn(M, N, Dim)
    arr[:, 0, :] = torch.zeros((M, Dim))
    for i in range(1, N):
        mu = arr[:, i - 1, :]
        arr[:, i, :] = arr[:, i - 1, :] + mu * dt + arr[:, i, :] * Gaussvar
    return arr

# generate the dataset
X = sample_NdFP(M, T, N, Dim)

class EigenFun(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack1 = nn.Linear(Dim, 6 * Dim)
        self.stack2 = nn.Linear(6 * Dim, 3 * Dim)
        self.stack3 = nn.Linear(3 * Dim, 1)
        self.cnst1 = nn.Linear(1, Dim)
        self.cnst2 = nn.Linear(Dim, 1)
        self.adj = nn.Linear(1, 1, bias=False)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        a = 0.875
        y = self.stack1(x)
        y = self.tanh(y)
        y = self.stack2(y)
        y = self.tanh(y)
        y = self.stack3(y)
        y = torch.squeeze(y)
        z = x.pow(2).sum(dim=1)
        w = 1. / (1 + z / (a ** 2))
        return y * w
    
    def lamda(self, lam):
        lam = self.cnst1(lam)
        lam = self.relu(lam).pow(9)
        lam = self.cnst2(lam)
        return lam

u = EigenFun()
lam = torch.tensor([2.])

c_shift = 2. * Dim

LRfrac = 1/150

# x_1 = torch.zeros((1, Dim))
# x_2 = torch.cat((torch.tensor([1.]), torch.zeros(Dim-1))).reshape(1, Dim)

exp1 = 1/4
exp2 = 1/8
exp3 = exp1 / (exp1 + exp2)

c_boundary = np.sqrt(Dim) ** np.sqrt(Dim)

n_epoch = 10000

dt = T / N
dBt = np.sqrt(dt)

lossarr = np.zeros(n_epoch)
lamdarr = np.zeros(n_epoch)
errarr = np.zeros(n_epoch)

for i in range(n_epoch):
    if i % 50 == 0:
        M_int = np.random.randint(int(M/200), int(M/25))
        rand_list = list(np.random.randint(M) for _ in range(M_int))
        X_rand = X[rand_list, :, :]
        X_new = torch.reshape(X_rand, (M_int * N, Dim))
        if i >= 500 and i % 500 == 0:
            LRfrac /= 2
        if i == 7500:
            LRfrac /= 2
        if i % 500 == 0:
            optimizer = torch.optim.Adamax(u.parameters(), lr=LRfrac)
    u_val = torch.reshape(u(X_new), (M_int, N))
#     X_val = X_rand[:, :-1, :].reshape((M_int*(N-1), Dim)).requires_grad_(True)
#     u_val2 = torch.reshape(u(X_val), (M_int, N-1))
    
    lamdas = u.lamda(lam)
#     lossy = torch.autograd.grad(u_val2, X_val, torch.ones_like(u_val2), retain_graph=True, create_graph=True)[0].reshape((M_int, N-1, Dim))
#     X_val = torch.reshape(X_val, (M_int, N-1, Dim))
#     lossx = torch.einsum('ijk, ijk -> ij', X_val, lossy)
#     loss1 = ((u_val[:, 2:] - u_val[:, :-2]) + (lamdas + Dim - c_shift / 2) * (u_val[:, :-2] / 2 + u_val[:, 1:-1] + u_val[:, 2:] / 2) * dt).sum(axis=0).pow(2).sum() / (M_int ** 2) / N
    loss1 = ((u_val[:, 3:] - u_val[:, :-3]) + (lamdas + Dim - c_shift / 2) * (u_val[:, :-3] / 2 + u_val[:, 1:-2] + u_val[:, 2:-1] + u_val[:, 3:] / 2) * dt).sum(axis=0).pow(2).sum() / (M_int ** 2) / N
    x_zeros = torch.zeros((1, Dim))
    loss2 = (u(x_zeros).reshape(1) - 12.5) ** 2
#     loss2 = (u2(x_rand1).reshape(1) + u2(x_rand2).reshape(1) + u2(x_rand3).reshape(1) - 30.) ** 2
#     loss2 = (torch.abs(u2(x_1).reshape(1)) + torch.abs(u2(x_2).reshape(1)) - 30.) ** 2
    
    loss = ((loss1 / dt) ** exp1 + (c_boundary * loss2) / (2 * Dim - 1) ** exp2) ** exp3
    loss.backward(retain_graph=True)

    lamd = lamdas.item()
    rvalu = 0.
    lossarr[i] = loss.item()
    lamdarr[i] = lamd
    errarr[i] = (lamd - c_shift / 2) / (c_shift / 2)
    if i % 100 == 0:
        print("round", i, ": loss=", loss.item(), "eigenvalue:", lamd, "eig. error:", abs(errarr[i]) * 100, "%")
    optimizer.step()
    optimizer.zero_grad()


np.savetxt('FPloss{0}x2ctrap'.format(Dim), lossarr)

np.savetxt('FPeigenval{0}x2ctrap'.format(Dim), lamdarr)

np.savetxt('FPerror{0}x2ctrap'.format(Dim), errarr)


arr = np.linspace(-np.pi, np.pi, 300)
x_arr = torch.stack([torch.linspace(-np.pi, np.pi, steps=300), ] * Dim).t()
V = (x_arr.cpu().detach().numpy() ** 2).sum(axis=1)
u_true = np.exp(-V).squeeze()
np.savetxt('FPtrue_u{0}x2ctrap'.format(Dim), u_true)

# Here we take alpha to be the ratio of our learned u and our true u at 0.
x0 = (np.zeros((Dim)) ** 2).sum()
alpha = u(torch.zeros((1, Dim))).cpu().detach().numpy().flatten() / (np.exp(-x0))

u_learned = u(x_arr).cpu().detach().numpy().flatten() / alpha
np.savetxt('FPlearned_u{0}x2ctrap'.format(Dim), u_learned)


err_val = u_learned - u_true
np.savetxt('FPerror_u{0}x2ctrap'.format(Dim), err_val)


error = 0.
arr = np.linspace(-np.pi, np.pi, 500)
torcharr = torch.linspace(-np.pi, np.pi, steps=500)
x_val = torch.stack([torcharr, ] * Dim).t()
uval = u(x_val).cpu().detach().numpy().flatten() / alpha
Vval = (x_val.cpu().detach().numpy() ** 2).sum(axis=1)
err = uval - np.exp(-Vval).squeeze()
maxerr = max(abs(err))
for k in range(500):
    err[k] = err[k] ** 2
    error += err[k]
error /= 500
print("L2 Error from -pi to pi:", np.sqrt(error))
print("Linf Error from -pi to pi:", maxerr)

