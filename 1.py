from datetime import date
print(date)
import pymc3 as pm
from random import seed
import matplotlib.pyplot as plt
from networkx.utils import cuthill_mckee_ordering
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
from parallel import parallel_generate_walks
from pymc3.distributions.mixture import NormalMixture
import numpy as np, pandas as pd, seaborn as sns
from precomputed_probabilities import _precompute_probabilities, _generate_walks
from Mixture_weights import Mixture_weights
from node2vec import Node2Vec
from numpy import linalg as LA
# Create a graph
#graph = nx.karate_club_graph()
seed(1)
Number_of_nodes = 10                                                                                                    # the graph nodes
p = 0.5                                                                                                                 # probability of connecting two nodes
embedding_dim = 128
sigma  = np.sqrt(10)
mu1 = np.zeros(embedding_dim)
iterations = 100
graph = nx.fast_gnp_random_graph(Number_of_nodes, p)                                                                    # generate the graph
nx.draw_networkx(graph, with_labels=True)                                                                               # draw the graph
rcm = list(cuthill_mckee_ordering(graph))
A = nx.adjacency_matrix(graph, nodelist=rcm)                                                                            # adjacency matrix of the graph
d_graph = _precompute_probabilities(graph)                                                                              # pre-computed probabilities
Mixture_w, neighbors  = Mixture_weights(d_graph, 0)                                                                              # to obtain Mixture weights, and neighbors
tao = 1.1
cov1 = sigma ** 2 * np.eye(embedding_dim)
EMBEDDING_FILENAME = '~/home/bigboss/PycharmProjects/HebbianGraphEmbedding/embeddings.emb'
EMBEDDING_MODEL_FILENAME = '~/home/bigboss/PycharmProjects/HebbianGraphEmbedding/embeddings.model'
embeddings = np.zeros((Number_of_nodes, embedding_dim))

#print(samp)
for i in range(Number_of_nodes):
    with pm.Model():
        embeddings[i, :] = pm.MvNormal.dist(mu=mu1, cov=cov1, shape=(embedding_dim, 1)).random()

def choicess(neighbors):
    l = np.arange(10)
    return np.delete(l, neighbors)

for m in range(iterations):
    cov1 = sigma ** 2 * np.eye(embedding_dim)

    for i in range(Number_of_nodes):
        wi = embeddings[i, :]
        Mixture_w, neighbors = Mixture_weights(d_graph, 0)
        # nagtive sampling
        neg_node = np.random.choice(choicess(neighbors), replace=False)
        wn = embeddings[neg_node,:]
        for idx, j in enumerate(neighbors):
            wj = embeddings[j,:]
            pij = Mixture_w[idx]
            wj = pm.MvNormal.dist(mu=wj, cov=cov1, shape=(embedding_dim, 1)).random()
            wi = wi + 0.001 * wj * pij - 0.001 * 0.5 * wn

        embeddings[i,:] = wi
    sigma = sigma/tao
def simialrity(embedding):
    n,m = embedding.shape
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i==0 and j==0:
                s[i][j] = 1
            x = embedding[i,:]
            y = embedding[j,:]
            s[i][j] = np.dot(x,y)/ (LA.norm(x) * LA.norm(y) )
    return s

s = simialrity(embeddings)
fig, ax = plt.subplots()
im = ax.imshow(s)
ax.set_xticklabels(np.arange(10))
ax.set_yticklabels(np.arange(10))
ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()
# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
#node2vec = Node2Vec(graph, dimensions=128, walk_length=5, num_walks=200, workers=4)  # Use temp_folder for big graphs
# Embed nodes
#model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
# Look for most similar nodes
#model.wv.most_similar('2')  # Output node names are always strings
# Save embeddings for later use
#model.wv.save_word2vec_format(EMBEDDING_FILENAME)
