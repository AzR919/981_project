import numpy as np
from math import comb
np.set_printoptions(precision=5, suppress=True, linewidth=150)

def WG_grpo(p, G, eps=0.0):
    tot = 0.0
    for S in range(G):
        pr = comb(G-1, S) * p**S * (1-p)**(G-1-S)
        rb1 = (S+1)/G; rb0 = S/G
        v1 = rb1*(1-rb1); v0 = rb0*(1-rb0)
        a1 = (1-rb1)/np.sqrt(v1+eps) if v1+eps > 0 else 0.0
        a0 = (-rb0)/np.sqrt(v0+eps) if v0+eps > 0 else 0.0
        tot += pr*(a1-a0)
    return tot
for G in [2,3,4,8,64,1024]:
    print(G, [round(WG_grpo(p,G),4) for p in [0.001,0.01,0.03,0.1,0.3,0.5,0.9,0.97]])
print('inf', [round(1/np.sqrt(p*(1-p)),4) for p in [0.001,0.01,0.03,0.1,0.3,0.5,0.9,0.97]])

def setup(X, c, W):
    logits = X @ W.T
    pi = np.exp(logits - logits.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    n, K = pi.shape
    p = pi[np.arange(n), c]
    E = -pi.copy(); E[np.arange(n), c] += 1
    g = (p[:,None]*E)[:,:,None] * X[:,None,:]   # n,K,d
    return p, g.reshape(n,-1)

X = np.array([[1,0.3],[1,0.5],[1,-0.1]])
c = np.array([1,1,0])
for a in [2.0,3.0,3.5,4.0,5.0]:
    W = np.array([[a,0],[0,0],[0,0.]])
    p, g = setup(X,c,W)
    kap = g@g.T
    J1 = g.mean(0)
    aa = g@J1
    res = {}
    for name, w in [('const',np.ones(3)),('grpo_inf',1/np.sqrt(p*(1-p))),('grpo8',np.array([WG_grpo(q,8) for q in p])),('grpo64',np.array([WG_grpo(q,64) for q in p])),('pass4',4*(1-p)**3),('pass10',10*(1-p)**9)]:
        res[name] = (w*aa).mean()
    print(a, p, 'a=',aa, {k: float(f"{v:.3e}") for k,v in res.items()})
    print(kap)
