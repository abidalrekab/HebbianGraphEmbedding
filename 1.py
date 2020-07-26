import pymc3 as pm
import networkx as kx

from pymc3.distributions.mixture import NormalMixture
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns



embedding_dim = 128
Number_of_nodes = 10
sigma  = np.sqrt(10)
mu1 = np.zeros(embedding_dim)
cov1 = sigma **2 * np.eye(embedding_dim)

with pm.Model():
    samp = pm.MvNormal.dist(mu=mu1, cov=cov1, shape=(embedding_dim, 1)).random()


print(samp)
for i in range(Number_of_nodes):
    pass
