import sys
from constants import *
from utils import *
from solvers import *
from problems import bound1, bound2

from scipy import sparse
from scipy.sparse import csr_array
from scipy.sparse import kron as spkron
from scipy.sparse import tril as sptril
from scipy.sparse.linalg import inv as spinv

def gen_a_list():
    al = []

    for n in grid_sizes:
        A = np.kron(np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 1), 1), -np.eye(n))
        D = -1.0 * np.diag(np.ones(n-1), -1) + -1.0 * np.diag(np.ones(n-1), 1) + 4 * np.diag(np.ones(n), 0)
        A += np.kron(np.diag(np.ones(n), 0), D)
        al.append(csr_array(A))
        
    return al
    
def gen_pr_lists():
    rl = []
    pl = []
    
    for ii,jj in zip(grid_sizes[:-1], grid_sizes[1:]):
        rl.append(csr_array(make_2d_pmat((ii,ii),(jj,jj))))
        pl.append(4.0*rl[-1])

    return rl, pl

if __name__ == '__main__':
    #parallel -j 8 "python run_expt.py {1} {2} 1{3}1 > results/{2}/{1}_{3}.txt" ::: {mg_1,mg_2,gs_1,sk_1,sk_2,lsk_1,lsk_2} ::: {prob1,prob2} ::: {1,2,3}
    solver_name = sys.argv[1]
    mu = int(solver_name.split("_")[1])
    solver_name = solver_name.split("_")[0]

    prob = sys.argv[2]
    seed = int(sys.argv[3])
    np.random.seed(seed)

    if prob == "prob1":
        gen_b = bound1
    elif prob == "prob2":
        gen_b = bound2

    b = gen_b(grid_sizes[-1])
    u = np.random.random(b.shape)
    
    if solver_name in ['gs', 'mg']:
        al = gen_a_list()
        rl, pl = gen_pr_lists()
        if solver_name == 'gs':
            solver = GaussSeidelSolver(al[-1])
        else:
            solver = MGSolver(al, rl, pl,mu=mu)

    elif solver_name in ['sk', 'lsk']:
        a1l = []
        p1l = []
        r1l = []
        
        for n in grid_sizes:
            A = 2 * np.diag(np.ones(n), 0) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
            a1l.append(sparse.csr_array(A))
        
        for ii,jj in zip(grid_sizes[:-1], grid_sizes[1:]):
            r1l.append(sparse.csr_array(make_1d_pmat(ii,jj)))
            p1l.append(2.0*r1l[-1])
        
        a2l = deepcopy(a1l)
        p2l = deepcopy(p1l)
        r2l = deepcopy(r1l)

        if solver_name =='sk':
            solver = SkeletalMGSolver([a1l, a2l], [r1l, r2l], [p1l, p2l],mu=mu)
        else:
            
            solver = LevelwiseSkeletalMGSolver([a1l, a2l], [r1l, r2l], [p1l, p2l],2*len(grid_sizes)-2, mu=1)
        
    twork = 0
    print(0, solver.residual_norm(u,b))
    while twork < work_max:
        u = solver.step(u, b)
        twork = solver.work()
        print(twork, solver.residual_norm(u,b))