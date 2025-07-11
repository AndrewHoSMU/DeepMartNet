#!/usr/bin/env python
# coding: utf-8

# A result for dimension 5. 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, vmap
from time import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import math, os, sys
print(torch.initial_seed())


# In[2]:


# dimension
D = 10

# domain width
L = 1.0


# In[3]:


# rhs

om = 2.

c = -1.

def f(x):
    return (-om ** 2 + c) * (om * x).cos().sum()


# boundary condition
def g(x):
    return (om * x).cos().sum()


def u_true(x):
    return (om * x).cos().sum()


# In[4]:


# path simulation time
T = 0.4

# Total umber of paths
M_tot = 100000

# Batch size of each epoch
M = 4000

# number of grid points
N = 100

# number of grid points at the boundaries
N_bdry = 10000

# weight of boundary loss
c_boundary = 1.0

# number of collocation points for evaluating the integral
N_test = 10000

# number of training epochs
n_epoch = 50000


# In[5]:


# time step size
dt = T / N
dBt = math.sqrt(dt)

def sample_NDstdBM_paths(M: int, D: int, T: float, N: int):
    return torch.cumsum(
        torch.concat([
            torch.zeros(M, 1, D),
            torch.randn(M, N, D) * math.sqrt(T / N)
        ], dim=1), dim=1
    )


W_tot = sample_NDstdBM_paths(M_tot, D, T, N)


# In[6]:


def mask_NDstdBM_paths(paths, L: float):
    oob = paths.abs().max(dim=2)[0].le(L).double()
    return oob.cummin(dim=1).values, oob.argmin(dim=1)


mask, exit_idx = mask_NDstdBM_paths(W_tot, L)
print(f"Boundary unreached rate: {mask[:, -1].sum().item() / M_tot * 100:.2f}%")


# In[7]:


def sample_boundary_points(N_bdry: int, D: int, L: float):
    rand_u = torch.rand(N_bdry, D) - 0.5
    return rand_u / rand_u.abs().max(dim=1)[0].unsqueeze(dim=1) * L


X_bdry = sample_boundary_points(N_bdry, D, L)


# In[8]:

'''
for N in [4, 8, 16, 32]:
    # time step size
    dt = T / N
    dBt = math.sqrt(dt)
    W_tot = sample_NDstdBM_paths(M_tot, D, T, N)
    mask, exit_idx = mask_NDstdBM_paths(W_tot, L)
    print(f"Boundary unreached rate: {mask[:, -1].sum().item() / M_tot * 100:.2f}%")
    X_bdry = sample_boundary_points(N_bdry, D, L)
    
    u = nn.Sequential(nn.Linear(D, 16),
                      nn.Tanh(), # nn.GELU(approximate="tanh"),
                      nn.Linear(16, 4),
                      nn.GELU(approximate="tanh"),
                      nn.Linear(4, 1))
    
    X_test = torch.rand(N_test, D) * 2 * L - L
    
    optimizer = torch.optim.Adamax(u.parameters(), lr=.05, )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    loss_rec = np.zeros(n_epoch + 1)
    l2_err_rec = np.zeros(n_epoch + 1)
    l2_norm_rec = np.zeros(n_epoch + 1)
    
    f_val_tot = vmap(vmap(f))(W_tot)
    g_val = vmap(g)(X_bdry)
    
    # get u0 by Feynman-Kac formula
    u0 = (vmap(g)(torch.stack([W_tot[i, exit_idx[i], :] for i in range(M_tot)])) * (c * exit_idx * dt / 2).exp() - .5 * \
         (f_val_tot * (torch.linspace(0, T, N+1) * c / 2).exp() * mask).sum(dim=1).mul(dt)).mean()
    u0.item()
    
    start_time = time()
    for i in range(n_epoch):
        batch_idx = torch.randint(0, M_tot, (M,))
        W = W_tot[batch_idx]
        f_val = f_val_tot[batch_idx]
        u_val = vmap(u)(W).squeeze()
        u_val_bdry = u(X_bdry).squeeze()
        loss_mart = ((u_val[:, 1:] - u_val[:, :-1] - (f_val - c * u_val)[:, :-1] / 2. * dt)
                     * mask[batch_idx, :-1]).sum(axis=0).pow(2).sum() / T / M ** 2
        # X_bdry = sample_boundary_points(N_bdry, D, L)
        loss_bdry = (u(X_bdry).squeeze() - g_val).pow(2).sum() / N_bdry
        # loss_0 = (u(torch.zeros(D)) - D) ** 2
        loss_0 = (u(torch.zeros(D)) - u0) ** 2
        loss = loss_mart + 0.01 * loss_bdry + 0. * loss_0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        u_test_val = u(X_test).squeeze()
        u_true_val = vmap(u_true)(X_test).squeeze()
        L2_loss = torch.sqrt((u_test_val - u_true_val).pow(2).sum() / N_test)
        L2_norm = torch.sqrt(u_test_val.pow(2).sum() / N_test)
    
        loss_rec[i] = loss.item()
        l2_err_rec[i] = L2_loss.item()
        l2_norm_rec[i] = L2_norm.item()
    
        if i % 100 == 0:
            scheduler.step()
    
        if i % 100 == 0 or i == n_epoch - 1:
            print(f"round {i}, loss={loss.item()}, L2 error={L2_loss.item()}, L2 norm={L2_norm.item()}")
    
    end_time = time()
    
    print(f"Training finished in {end_time - start_time} seconds.")
    print("Now saving the results ... ", end="")
    torch.save(u.state_dict(), os.path.join(".", f"the_network_N_{N}_M_{M}.pt"))
    np.savez(f"loss_and_err_N_{N}_M_{M}.npz", loss_rec=loss_rec, l2_err_rec=l2_err_rec)
    print("done.")
    print("Congratulations! Everything is done.")
'''

# In[ ]:


for i in [4, 8, 16, 32]:
    rec = np.load(f"loss_and_err_N_{i}_M_4000.npz")
    l2_err_rec = rec["l2_err_rec"]
    loss_rec = rec["loss_rec"]

    # plt.semilogy(loss_rec[:-1])
    # plt.title("loss")
    # plt.show()
    
    plt.semilogy(l2_err_rec[:-1])
plt.title("error")
plt.legend(["Δt=0.2", 
            "Δt=0.1", 
            "Δt=0.05", 
            "Δt=0.025", 
            "Δt=0.0125"])
plt.show()


for i in [4, 8, 16, 32]:
    rec = np.load(f"loss_and_err_N_{i}_M_4000.npz")
    l2_err_rec = rec["l2_err_rec"]
    loss_rec = rec["loss_rec"]

    plt.semilogy(loss_rec[:-1])
plt.title("loss")
plt.legend(["Δt=0.2", 
            "Δt=0.1", 
            "Δt=0.05", 
            "Δt=0.025", 
            "Δt=0.0125"])
plt.show()
    
    # plt.semilogy(l2_err_rec[:-1])
    # plt.title("error")
    # plt.show()


# In[ ]:


errs = []
for i in [4, 8, 16, 32]:
    rec = np.load(f"loss_and_err_N_{i}_M_4000.npz")
    l2_err_rec = rec["l2_err_rec"]
    loss_rec = rec["loss_rec"]

    # plt.semilogy(loss_rec[:-1])
    # plt.title("loss")
    # plt.show()
    errs.append(l2_err_rec[-2])
errs


# In[ ]:


plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
def regress_plot(x, y, xlabel="-log10(Δt)", ylabel="-log10(err)", title=""):
    from sklearn.linear_model import LinearRegression
    
    # Reshape x for sklearn
    X = x.reshape(-1, 1)
    
    # Create the LTS regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict using the model
    y_pred = model.predict(X)
    
    # Scatter plot for the dataset
    plt.scatter(x, y, color='blue', label='Data points')
    
    # Plot the LTS regression line
    plt.plot(x, y_pred, color='red', linewidth=2, label='LTS Regression Line')

    # Get the coefficients of the linear regression 
    coef = model.coef_[0] 
    intercept = model.intercept_
    
    # Add the regression equation to the plot 
    equation_text = f'-log10(err) = {coef:.2f} (-log10(Δt)) + {intercept:.2f}'
    plt.text(0.4, 0.35, equation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    # Add labels and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    # plt.legend()
    
    # Show the plot
    plt.show()


# In[ ]:


regress_plot(-np.log10(np.array([0.1, 0.05, 0.025, 0.0125])),
             -np.log10(np.array(errs)))

# In[ ]:





# In[ ]:


1 + 1


# In[ ]:




