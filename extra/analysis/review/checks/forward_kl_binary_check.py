import math
from math import comb, sqrt, log
def WG(p,G):
    # GRPO std-normalised advantage, population std, 0/0 := 0
    e1=0.0; e0=0.0
    for k in range(G):  # k = successes among the other G-1
        w=comb(G-1,k)*p**k*(1-p)**(G-1-k)
        K=k+1  # sample is correct
        if 0<K<G: e1+=w*sqrt((G-K)/K)
        K=k    # sample is wrong
        if 0<K<G: e0+=w*(-sqrt(K/(G-K)))
    return e1-e0
for G in (2,4,8,16,64):
    print(G,[round(WG(p,G),4) for p in (1e-6,0.1,0.5,0.9,1-1e-6)], 'sqrt(G-1)=',round(sqrt(G-1),4),'inf-G at .5:',2.0)
logit=lambda p: log(p/(1-p))
# GX-Chen forward-KL closed form projected onto binary reward: solve for Lambda by bisection
def fwd(pref,beta):
    lo,hi=1+1e-15,1e9
    f=lambda L: beta*pref/(L-1)+beta*(1-pref)/L-1
    for _ in range(300):
        mid=(lo+hi)/2
        if f(mid)>0: lo=mid
        else: hi=mid
    L=(lo+hi)/2
    return beta*pref/(L-1)
for pref,beta in ((0.2,0.5),(0.05,0.1),(0.6,2.0)):
    p=fwd(pref,beta)
    quad=0.5*((1-beta)+sqrt((1-beta)**2+4*beta*pref))   # Vojnovic eq (14) with gamma=1
    print('fwd',pref,beta,'p*=',round(p,6),'(p-pref)/(p(1-p))=',round((p-pref)/(p*(1-p)),6),'1/beta=',1/beta,'Vojnovic(14)=',round(quad,6))
    prev=1/(1+ (1-pref)/pref*math.exp(-1/beta))
    print('rev',pref,beta,'p*=',round(prev,6),'logit diff=',round(logit(prev)-logit(pref),6))
# Vojnovic (15) vs project forward-KL formula with W_inf
for pref,beta in ((0.2,0.5),(0.05,2.0)):
    p=(2*beta**2*pref+1+sqrt(1+4*beta**2*pref*(1-pref)))/(2*(1+beta**2))
    print('V15',pref,beta,round(p,6),'lhs',round((p-pref)/(p*(1-p)),6),'W_inf/beta',round(1/(beta*sqrt(p*(1-p))),6))
# reverse-KL (project naming) at G=inf: number of interior roots of logit p - logit pref - 1/(beta sqrt(p(1-p)))
def roots(pref,beta,W):
    xs=[i/20000 for i in range(1,20000)]
    g=[logit(x)-logit(pref)-W(x)/beta for x in xs]
    return [round(xs[i],4) for i in range(len(xs)-1) if g[i]*g[i+1]<0]
for beta in (0.5,1,2,3,5):
    print('beta',beta,'G=inf roots',roots(0.2,beta,lambda x:1/sqrt(x*(1-x))),'G=8 roots',roots(0.2,beta,lambda x:WG(x,8)))
