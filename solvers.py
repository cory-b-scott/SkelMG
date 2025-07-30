import numpy as np

from utils import *

from scipy import sparse
from scipy.sparse import tril as sptril
from scipy.sparse import kron as spkron
from scipy.sparse.linalg import inv as spinv
from scipy.sparse.linalg import svds as spsvds

from copy import deepcopy

class Solver():

    def __init__(self):
        pass

    def residual(self, u, b):
        return b - (self.A @ u)

    def residual_norm(self, u, b):
        return np.linalg.norm(self.residual(u, b), ord=np.inf)

class GaussSeidelSolver(Solver):

    def __init__(self, A, static=True):
        self.cost = 0
        self.A = A
        self.static=static
        self.E = sptril(self.A)
        self.Einv = spinv(self.E)
        #self.cost += A.shape[0]**3

    def step(self, u, b):
        #print(u,b,self.A)
        if self.A.shape[0] == 1:
            r = b - (self.A * u).todense()
        else:
            r = b - self.A@u
        self.cost += self.A.count_nonzero()
        #self.cost += 2*(self.A.shape[0]**2)
        if not self.static:
            self.E = sptril(self.A)
            self.Einv = spinv(self.E)
            #self.cost += A.shape[0]**3
        #print(u.shape, b.shape, self.residual_norm(u,b))
        return u + self.Einv@r

    def work(self):
        return self.cost

    def err_prop_matrix(self):
        if self.A.shape[0] == 1:
            return np.zeros((self.A.shape[0],self.A.shape[0]))
        return sparse.eye_array(self.A.shape[0]) - self.Einv.dot(self.A)

class MGSolver(Solver):
    def __init__(self, Alist, Plist, Rlist, mu=1):
        self.mu=mu
        self.A = Alist[-1]
        self.smoother = GaussSeidelSolver(self.A)
        self.cost = 0
        if len(Alist) == 1:
            self.P = sparse.eye_array(self.A.shape[0])
            self.R = sparse.eye_array(self.A.shape[0])
            self.subsolver = None
        else:
            self.P = Plist[-1]
            self.R = Rlist[-1]
            self.subsolver = MGSolver(Alist[:-1], Plist[:-1], Rlist[:-1],mu=mu)

    def step(self, u, b):
        v = self.smoother.step(u, b)
        if self.subsolver is None:
            return v
        r = self.smoother.residual(v, b)
        rr = restrict(self.R, r)
        v2 = np.zeros_like(rr)
        for i in range(self.mu):
            v2 = self.subsolver.step(v2, rr)
        #print(self.P.shape, v2.shape)
        rrr = prolong(self.P, v2)
        #self.cost += self.A.shape[0]**2
        #print(rrr.shape)
        v = self.smoother.step(v + rrr, b)
        return v

    def work(self):
        if self.subsolver is None:
            return self.smoother.work()
        else:
            return self.subsolver.work() + self.smoother.work() + self.cost

    def err_prop_matrix(self):
        S = self.smoother.err_prop_matrix()
        if self.subsolver is None:
            return S
        else:
            Minner = self.subsolver.err_prop_matrix()
            Minner = sparse.eye_array(Minner.shape[0]) - Minner
            #print( spinv(self.subsolver.A).shape,  self.R.shape, self.smoother.A.shape )
            post = (spinv(self.subsolver.A) @ self.R) @ self.smoother.A
            if len(post.shape) == 1:
                post = post.reshape(1,-1)
            #print(self.P.T.shape, Minner.shape, post.shape)
            Minner = (self.P.T @ (Minner @ post))
            Minner = sparse.eye_array(Minner.shape[0]) - Minner
            return S.T @ (Minner @ S)

class SkeletalMGSolver(Solver):
    def __init__(self, Alists, Plists, Rlists, mu=1):
        self.mu=mu
        self.dim = len(Alists)
        self.Amats = []
        self.subprobs = []
        self.Pmats = []
        self.Rmats = []
        for i in range(self.dim):
            if len(Alists[i]) > 0:
                self.Amats.append(Alists[i][-1])
            if len(Alists[i]) > 1:
                smallA = deepcopy(Alists)
                smallP = deepcopy(Plists)
                smallR = deepcopy(Rlists)
                smallA[i].pop(-1)
                Plist_temp = [
                    (sparse.eye_array(Alists[j][-1].shape[0]) if len(Alists[j])> 0 else sparse.eye_array(1) )
                for j in range(self.dim)]
                Rlist_temp = [
                    (sparse.eye_array(Alists[j][-1].shape[0]) if len(Alists[j])> 0 else sparse.eye_array(1) )
                for j in range(self.dim)]
                Plist_temp[i] = smallP[i].pop(-1)
                Rlist_temp[i] = smallR[i].pop(-1)
                self.Pmats.append(kron_from_list(Plist_temp))
                self.Rmats.append(kron_from_list(Rlist_temp))
                self.subprobs.append(SkeletalMGSolver(smallA, smallP, smallR, mu=mu))
        self.A = kronsum_from_list(self.Amats)
        self.smoother = GaussSeidelSolver(self.A)
        #print("&&&")
        #print(self.A.shape)
        #print([item.A.shape for item in self.subprobs])
        #print([item.shape for item in self.Pmats])
        #print([item.shape for item in self.Rmats])
        self.cost = 0

    def step(self, u, b):
        v = self.smoother.step(u, b)
        if len(self.subprobs) == 0:
            return v
        corr = np.zeros_like(v)
        r = self.smoother.residual(v, b)
        for i,(P,R,sub) in enumerate(zip(self.Pmats, self.Rmats, self.subprobs)):
            rr = restrict(R, r)
            v2 = np.zeros_like(rr)
            for j in range(self.mu):
                v2 = sub.step(v2, rr)
            pv2 = prolong(P, v2)
            corr += pv2
            #for j,(Pp,Rp) in enumerate(zip(self.Pmats, self.Rmats)):
            #    if j != i:
            #        corr -= (1/len(self.subprobs)) * prolong(Pp, restrict(Rp, pv2))
            #self.cost += self.A.shape[0]**2
        #print(rrr.shape)
        v = self.smoother.step(v + corr, b)
        return v

    def work(self):
        return self.smoother.work() + self.cost + sum([item.work() for item in self.subprobs])

def get_tuples(length, total):
    if length == 1:
        yield (total,)
        return

    for i in range(total + 1):
        for t in get_tuples(length - 1, total - i):
            yield (i,) + t

def get_tuples_limited(length, total, maxes):
    if length == 1:
        yield (min(total, maxes[0]),)
        return

    for i in range(min(total, maxes[0]) + 1):
        for t in get_tuples_limited(length - 1, total - i, maxes[1:]):
            newt = (i,) + t
            if sum(newt) == total:
                yield newt

class LevelwiseSkeletalMGSolver(MGSolver):
    def __init__(self, Alists, Plists, Rlists, L, mu=1):
        self.mu=mu
        self.dim = len(Alists)
        self.smoother = None
        maxes = tuple([len(item)-1 for item in Alists])
        tups = list(get_tuples_limited(self.dim, L, maxes))
        sub_tups = list(get_tuples_limited(self.dim, L-1, maxes))

        self.subsolver = None

        #print(list(tups))
        #print(list(sub_tups))

        alist = []
        for tup in tups:
            asel = [Alists[dim_idx][tup_idx] for dim_idx,tup_idx in enumerate(tup)]
            newA = kronsum_from_list(asel)
            alist.append(newA)
        self.A = sparse.block_diag(alist)
        self.smoother = GaussSeidelSolver(self.A)

        pm_blocks = [[None for i in range(len(sub_tups))] for j in range(len(tups))]
        rm_blocks = [[None for i in range(len(tups))] for j in range(len(sub_tups))]
        rm_occupancy = [[0 for i in range(len(tups))] for j in range(len(sub_tups))]

        if L > 0:
            for i,tu in enumerate(tups):
                for j,stu in enumerate(sub_tups):
                    diff = [abs(u-v) for u,v in zip(tu, stu)]
                    if sum(diff) == 1:
                        pm_mats = []
                        rm_mats = []
                        for di, (idx1, idx2) in enumerate(zip(tu,stu)):
                            #print(di, len(Plists[di]), idx1, idx2)
                            if idx1 - idx2 == 1:
                                pm_mats.append(Plists[di][idx2].T)
                                rm_mats.append(Rlists[di][idx2])
                            else:
                                pm_mats.append(sparse.eye_array(Alists[di][idx2].shape[0]))
                                rm_mats.append(sparse.eye_array(Alists[di][idx2].shape[0]))
                        pm_blocks[i][j] = kron_from_list(pm_mats)
                        rm_blocks[j][i] = kron_from_list(rm_mats)
                        rm_occupancy[j][i] += 1

            #for i,tu in enumerate(tups):
            #    for j,stu in enumerate(sub_tups):
            #        if rm_blocks[j][i] is not None:
            #            rm_blocks[j][i] /= max(rm_occupancy[j][i], 1)

            self.R = sparse.block_array(rm_blocks)


            self.P = sparse.block_array(pm_blocks)
            self.P = self.P.T

        
            self.subsolver = LevelwiseSkeletalMGSolver(Alists, Plists, Rlists, L-1, mu=mu)
        self.cost = 0




        return
        for i in range(self.dim):
            if len(Alists[i]) > 0:
                self.Amats.append(Alists[i][-1])
            if len(Alists[i]) > 1:
                smallA = deepcopy(Alists)
                smallP = deepcopy(Plists)
                smallR = deepcopy(Rlists)
                smallA[i].pop(-1)
                Plist_temp = [
                    (sparse.eye_array(Alists[j][-1].shape[0]) if len(Alists[j])> 0 else sparse.eye_array(1) )
                for j in range(self.dim)]
                Rlist_temp = [
                    (sparse.eye_array(Alists[j][-1].shape[0]) if len(Alists[j])> 0 else sparse.eye_array(1) )
                for j in range(self.dim)]
                Plist_temp[i] = smallP[i].pop(-1)
                Rlist_temp[i] = smallR[i].pop(-1)
                self.Pmats.append(kron_from_list(Plist_temp))
                self.Rmats.append(kron_from_list(Rlist_temp))
                self.subprobs.append(SkeletalMGSolver(smallA, smallP, smallR, mu=mu))
        self.A = kronsum_from_list(self.Amats)
        self.smoother = GaussSeidelSolver(self.A)
        #print("&&&")
        #print(self.A.shape)
        #print([item.A.shape for item in self.subprobs])
        #print([item.shape for item in self.Pmats])
        #print([item.shape for item in self.Rmats])
        self.cost = 0
