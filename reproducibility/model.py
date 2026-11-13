"""Models and deterministic frequency calculations for T-ITS-26-09-5185 R1.
Speeds m/s, net spacings m, time s. First letter in link type is predecessor.
"""
import numpy as np
from scipy.optimize import minimize_scalar

DEFAULT=dict(alpha=.6,beta=.6,kp=.2,kv=.7,ka=1.,hcc=.6)

def links(w,vp,hac,beta=.6,alpha=.6,kp=.2,kv=.7,ka=1.,hcc=.6,c=1.,delta=0.,tau=0.,feedback=0.):
    s=1j*np.asarray(w); a=alpha*vp
    # HDV reaction delay applies to all terms of the acceleration law.
    e=np.exp(-tau*s)
    hh=e*(beta*s+a)/(s*s+e*((alpha+beta)*s+a))
    ac=alpha*vp*c
    ch=e*(beta*s+ac)/(s*s+e*((alpha+beta)*s+ac))
    ef=np.exp(-feedback*s)
    cc=(ka*np.exp(-delta*s)*s*s+ef*(kv*s+kp))/(s*s+ef*((kv+kp*hcc)*s+kp))
    hc=ef*(kv*s+kp)/(s*s+ef*((kv+kp*hac)*s+kp))
    return dict(CC=cc,HC=hc,CH=ch,HH=hh)

def ring_types(n,k,b):
    assert 0<k<n and 1<=b<=min(k,n-k)
    cs=[k//b+(i<k%b) for i in range(b)]
    hs=[(n-k)//b+(i<(n-k)%b) for i in range(b)]
    return ''.join('C'*x+'H'*y for x,y in zip(cs,hs))

def counts(types,ring=False):
    return {q:sum(types[i-1]+types[i]==q for i in range(0 if ring else 1,len(types))) for q in ['CC','HC','CH','HH']}

def product(w,types,vp,hac,ring=False,**kw):
    gs=links(w,vp,hac,**kw); ct=counts(types,ring)
    return np.prod([gs[t]**ct[t] for t in gs],axis=0)

def refined_peak(fun,lo=.1,hi=10.,samples=1800):
    w=np.geomspace(lo,hi,samples); a=np.abs(fun(w)); cand=[(float(a[0]),lo),(float(a[-1]),hi)]
    for j in np.flatnonzero((a[1:-1]>=a[:-2])&(a[1:-1]>=a[2:]))+1:
        r=minimize_scalar(lambda x:-float(abs(fun(np.exp(x)))),bounds=(np.log(w[j-1]),np.log(w[j+1])),method='bounded',options={'xatol':1e-12})
        cand.append((-float(r.fun),float(np.exp(r.x))))
    return max(cand)

def matrices(types,vp,hac,ring=False,topology='PF',gain=.8,normalized=False,**kw):
    p=DEFAULT|kw; n=len(types); off=0 if ring else 1; m=n-off
    # Descriptor: E a = K xi + V eta + b_v eta_lead.
    E=np.eye(m); K=np.zeros((m,m)); V=np.zeros((m,m)); bv=np.zeros(m); ba=np.zeros(m)
    Q=np.zeros((m,m)); qv=np.zeros(m)
    def add(row,j,val):
        if not ring and j==0: bv[row]+=val
        else: V[row,j-off]+=val
    for i in range(off,n):
        r=i-off; pred=(i-1)%n
        Q[r,r]-=1
        if not ring and pred==0: qv[r]+=1
        else: Q[r,pred-off]+=1
        if types[i]=='H':
            K[r,r]=p['alpha']*vp; V[r,r]=-p['alpha']-p['beta']; add(r,pred,p['beta'])
        else:
            coop=types[pred]=='C'; h=p['hcc'] if coop else hac
            K[r,r]=p['kp']; V[r,r]=-p['kv']-p['kp']*h; add(r,pred,p['kv'])
            if coop:
                if not ring and pred==0: ba[r]+=p['ka']
                else:E[r,pred-off]-=p['ka']
            extra=[]
            if topology=='PLF' and i>1: extra=[0] # instrumented exogenous leader broadcasts speed
            if topology in ['2PF','MPF']:
                extra=[j for j in range(max(0,i-(2 if topology=='2PF' else 3)),i-1) if types[j]=='C']
            if topology=='BD' and i+1<n and types[i+1]=='C': extra=[i+1]
            g=gain/max(1,len(extra)) if normalized else gain
            for j in extra: V[r,r]-=g; add(r,j,g)
    Ei=np.linalg.inv(E)
    A=np.block([[np.zeros((m,m)),Q],[Ei@K,Ei@V]])
    B0=np.r_[qv,Ei@bv]; B1=np.r_[np.zeros(m),Ei@ba]
    return A,B0,B1

def spectral_abscissa(types,vp,hac,**kw):
    A,_,_=matrices(types,vp,hac,ring=True,**kw)
    eig=np.linalg.eigvals(A); j=np.argmin(abs(eig))
    assert abs(eig[j])<1e-7
    return float(np.max(np.delete(eig,j).real))

def response(types,w,vp=1.2,hac=1.5,**kw):
    A,B0,B1=matrices(types,vp,hac,**kw); s=1j*np.asarray(w)
    rhs=B0+s[...,None]*B1
    x=np.linalg.solve(s[...,None,None]*np.eye(len(A))-A,rhs[...,None])[...,0]
    return x[...,-1],float(max(np.linalg.eigvals(A).real))
