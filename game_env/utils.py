import numpy as np

def norm_adj(adj):
    adj += np.eye(adj.shape[0])
    degr = np.array(adj.sum(1))
    degr = np.diag(np.power(degr, -0.5))
    return degr.dot(adj).dot(degr)