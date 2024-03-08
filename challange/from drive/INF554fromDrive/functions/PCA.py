

import numpy as np
from sklearn.decomposition import PCA
import torch

def rewrite_features(graph,mu=None,sigma=None,pca=None,device="cpu",n_components=100,keep_out_features=2):
    X = np.array(graph.x.cpu().numpy())
    x = X[:,:keep_out_features]
    X = X[:,keep_out_features:]
    if mu is None:
        pca = PCA(n_components=n_components)
        Y = pca.fit_transform(X)
        X = np.hstack((x,Y))
        mu = np.mean(X,axis=0)
        sigma = np.std(X,axis=0)
    else:
        Y = pca.transform(X)
        X = np.hstack((x,Y))
    X = (X-mu)
    X = X/sigma
    graph.x = torch.tensor(X.tolist(), dtype=torch.float).to(device)

    return graph, mu, sigma, pca
    
