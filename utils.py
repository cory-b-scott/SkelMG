import numpy as np

from scipy import sparse
from scipy.sparse import kron as spkron

def prolong(P, vec):
    return (P.T) @ vec

def restrict(P, vec):
    #print(P.shape, vec.shape)
    return P @ vec

def make_2d_pmat(s1, s2):
    (x1,y1) = s1
    (x2,y2) = s2
    pl = np.kron( np.kron( np.eye(x1) , np.ones((x2//x1,1)) ), np.kron( np.eye(y1) , np.ones((y2//y1,1))  ) )
    #print(pl.shape)
    #quit()
    #print(pl)
    pl /= pl.sum(0)
    return pl.T

def make_1d_pmat(x1, x2):
    pl = np.kron( np.eye(x1) , np.ones((x2//x1,1)) )
    #print(pl.shape)
    #quit()
    #print(pl)
    pl /= pl.sum(0)
    return pl.T

def kron_from_list(mat_list):
    m = np.ones((1,1))
    #print(*mat_list)
    for mat in mat_list:
        m = spkron(m, mat)
    #print(m.shape)
    return m

def kronsum_from_list(mat_list):
    Am = 0*kron_from_list(mat_list)
    for i in range(len(mat_list)):
        Am += kron_from_list([(mat_list[j] if i == j else np.eye(mat_list[j].shape[0])) for j in range(len(mat_list))])
    return Am