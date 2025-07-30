
import numpy as np

def bound1(n1, n2=None):
    if n2 == None:
        n2 = n1
    f = np.zeros((n1, n2))
    f[::, 0] += 10
    f[::, -1] -= 10
    f[-1, ::] += 10
    f[0, ::] -= 10
    bb = f.flatten()
    return bb

def bound2(n1, n2=None):
    if n2 == None:
        n2 = n1
    f = np.zeros((n1, n2))
    f[:n1//2, 0] += 10
    f[n1//2:, 0] -= 10
    f[:n1//2, -1] -= 10
    f[n1//2:, -1] += 10
    f[0,:n2//2] -= 10
    f[0,n2//2:] += 10
    f[-1, :n2//2] += 10
    f[-1, n2//2:] -= 10
    bb = f.flatten()
    return bb

