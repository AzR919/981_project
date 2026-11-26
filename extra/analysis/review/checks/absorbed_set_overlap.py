import pandas as pd, numpy as np
r=pd.read_csv('results/stretch/cache/runs_s_drift.csv')
z=np.load('results/stretch/cache/prompts_s_drift.npz')
pos={int(v):i for i,v in enumerate(z['row_train'])}
def absset(i):
    j=pos[i]; return (z['p_final'][j]<0.01)&(z['p_init'][j]>=0.01)
m=r[(r.block=='main')]
out=[]
for lvl in ['c09','c05','m64','lin']:
  for (eta,seed),g in m[m.level==lvl].groupby(['eta','seed']):
    d={u:absset(i) for u,i in zip(g['update'],g.index)}
    a,b=d['sampled'],d['expected_full']
    if a.sum()+b.sum()>=10:
        out.append((lvl,eta,seed,a.sum(),b.sum(),(a&b).sum()/max((a|b).sum(),1)))
o=pd.DataFrame(out,columns=['lvl','eta','seed','n_samp','n_full','jacc'])
print(o.groupby('lvl').jacc.describe().round(3))
c=m[(m.level=='c09')&(m['update']=='expected_full')]
for seed,g in c.groupby('seed'):
    sets=[absset(i) for i in g.sort_values('eta').index]
    ref=sets[6]
    print(seed,[round((s&ref).sum()/max((s|ref).sum(),1),2) for s in sets])
