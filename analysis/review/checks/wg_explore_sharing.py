import numpy as np
from wg_explore import setup, WG_grpo
np.set_printoptions(precision=5, suppress=True, linewidth=160)
X = np.array([[1,0.3],[1,0.5],[1,-0.1]])
c = np.array([1,1,0])
def weights(p):
    return {'const':np.ones_like(p),'grpo4':np.array([WG_grpo(q,4) for q in p]),'grpo8':np.array([WG_grpo(q,8) for q in p]),'grpo64':np.array([WG_grpo(q,64) for q in p]),'grpo_inf':1/np.sqrt(p*(1-p)),'pass4':4*(1-p)**3,'pass10':10*(1-p)**9}
for pe, ph in [(0.9,0.03),(0.9,0.02),(0.85,0.03),(0.8,0.04),(0.8,0.03),(0.7,0.05),(0.6,0.05),(0.5,0.05),(0.5,0.03)]:
    p2 = 1-pe-ph
    W = np.array([[np.log(pe/p2),0],[np.log(ph/p2),0],[0,0.]])
    p,g = setup(X,c,W)
    J1 = g.mean(0); aa = g@J1
    out = {k: float(f"{(w*aa).mean():.3e}") for k,w in weights(p).items()}
    print(pe,ph,'p=',p,'a=',aa, out)
