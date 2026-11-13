"""Run frequency, ring, topology, gain, delay and composition studies."""
from pathlib import Path
import csv,json,itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import expm
from numpy.polynomial import polynomial as poly
from model import *
OUT=Path(__file__).parent/'results'; OUT.mkdir(exist_ok=True)
W=np.geomspace(.1,10,1800)
def save(name,rows):
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(name,len(rows),flush=True)

def main():
    grid=[]
    for vp in np.round(np.arange(.2,2.001,.05),3):
        for h in np.round(np.arange(.2,3.001,.05),3):
            gs=links(W,vp,h); vals=[]; eig=[]
            for b in range(1,5):
                vals.append(refined_peak(lambda w:product(w,ring_types(12,4,b),vp,h,True))[0])
                eig.append(spectral_abscissa(ring_types(12,4,b),vp,h))
            f=np.log(vals); valid=[b for b in range(1,5) if eig[b-1]<-1e-8]
            grid.append(dict(vp=vp,hac=h,physical=int(h>.6),best_b=int(np.argmin(vals)+1),stable_best_b=min(valid,key=lambda b:vals[b-1]) if valid else 0,
                convex_error=float(min(np.diff(f,2))),**{f'theta{b}':vals[b-1] for b in range(1,5)},**{f'abscissa{b}':eig[b-1] for b in range(1,5)}))
    save('ring_grid',grid)
    cases=[]
    for vp,kv,h in itertools.product([.4,.8,1.2,1.6,2.],[.4,.7,1.],[.8,1.,1.2,1.5,2.,2.5]):
        a=.6*vp; A=2.25-2*a; B=(kv+.2*h)**2-.4
        c2=.81-kv*kv; c1=.81*B+a*a-kv*kv*A-.04; c0=.04*(.81-A)+a*a*(B-kv*kv)
        hd=(np.sqrt(max(0,kv*kv+.4-.04*(.81-A)/a**2))-kv)/.2
        def f(w,other):
            g=links(w,vp,h,beta=.9,kv=kv)
            return g['CC']**2*g['HH']**4*g['HC' if other else 'HH']
        th,wh=refined_peak(lambda w:f(w,False)); ti,wi=refined_peak(lambda w:f(w,True))
        admissible=c2>1e-12 and c1>1e-12
        cases.append(dict(vp=vp,kv=kv,hac=h,hdom=hd,c2=c2,c1=c1,c0=c0,admissible=int(admissible),head=th,other=ti,head_w=wh,other_w=wi,converse_failure=int(admissible and h<hd and ti<th-1e-10)))
    save('endpoint_90',cases)
    cells=[]
    for vp,h in itertools.product([.6,1.2,1.6],[.5,1.,1.5,2.,2.5]):
        p=[product(W,ring_types(12,4,b),vp,h,True) for b in range(1,5)]
        pk=[max(abs(x)) for x in p]; en=[np.trapezoid(abs(x)**2,W) for x in p]
        cells.append(dict(vp=vp,hac=h,peak_b=int(np.argmin(pk)+1),energy_b=int(np.argmin(en)+1)))
    save('index_15',cells)
    # Continuous-frequency refinement and actual band changes.
    refinement=[]
    for vp,h in itertools.product([.6,1.2,1.6,1.9],[.8,1.2,1.5,1.6,1.7,2.]):
        vals=[refined_peak(lambda w:product(w,ring_types(12,4,b),vp,h,True)) for b in range(1,5)]
        refinement.append(dict(vp=vp,hac=h,best_b=int(np.argmin([v[0] for v in vals])+1),**{f'theta{b}':vals[b-1][0] for b in range(1,5)}))
    save('peak_refinement',refinement)
    bands=[]
    for lo,hi in [(.05,10),(.1,2),(.1,20)]:
        ww=np.geomspace(lo,hi,2400)
        changed=0
        for row in grid:
            g=links(ww,row['vp'],row['hac']); v=[max(abs(g['CC']**(4-b)*g['HH']**8*g['HC']**b)) for b in range(1,5)]
            changed+=int(np.argmin(v)+1!=row['best_b'])
        bands.append(dict(lo=lo,hi=hi,changed=changed,total=len(grid)))
    save('band_sensitivity',bands)
    gains=[]
    for ka,hcc,kp,kv in itertools.product([.5,.8,1,1.2],[.1,.2,.6,1],[.1,.2,.3],[.35,.7,1.05]):
        kw=dict(ka=ka,hcc=hcc,kp=kp,kv=kv)
        for vp,h in itertools.product([1.2,1.9],[.8,1.6,2.]):
            ps=[product(W,ring_types(12,4,b),vp,h,True,**kw) for b in range(1,5)]
            vals=[max(abs(p)) for p in ps]
            gains.append(dict(**kw,vp=vp,hac=h,cc_stable=int(ka<=1 and 2*kv*hcc+kp*hcc*hcc>=2*(1-ka)),best_b=int(np.argmin(vals)+1),regret_b1=float(vals[0]/min(vals)-1),regret_b4=float(vals[-1]/min(vals)-1)))
    save('gain_sensitivity',gains)
    # All 56 open arrangements. Nonlocal links use CAV speed broadcasts only,
    # except PLF, whose external leader is instrumented independently of its label.
    top=[]; ww=np.geomspace(.1,10,500)
    for topology,gain,norm in itertools.product(['PF','PLF','2PF','MPF','BD'],[0,.2,.5,.8,1.2],[False,True]):
        for cc in itertools.combinations(range(8),3):
            types=''.join('C' if i in cc else 'H' for i in range(8))
            r,e=response(types,ww,beta=.9,topology=topology,gain=gain,normalized=norm)
            ct=counts(types)
            top.append(dict(topology=topology,gain=gain,normalized=int(norm),types=types,theta=float(max(abs(r))),abscissa=e,**ct))
    save('topologies',top)
    # Delay response is an input-output study; it is not a DDE pole certificate.
    delays=[]
    for channel,values in [('delta',[0,.1,.2,.5]),('tau',[0,.2,.4,.6,.8]),('feedback',[0,.05,.1,.2])]:
        for delay in values:
            for vp,h in itertools.product([1.2,1.9],[.8,1.2,1.6,1.7,2.]):
                ps=[product(W,ring_types(12,4,b),vp,h,True,**{channel:delay}) for b in range(1,5)]
                v=[max(abs(p)) for p in ps]
                delays.append(dict(channel=channel,delay=delay,vp=vp,hac=h,best_b=int(np.argmin(v)+1),**{f'theta{b}':float(v[b-1]) for b in range(1,5)}))
    save('delays',delays)
    # Exact invariance of ring poles, characteristic polynomial, and transient decay.
    checks=[]
    for vp,h in [(.6,2.),(1.9,.8),(1.2,2.)]:
        t=ring_types(12,4,2); A,_,_=matrices(t,vp,h,ring=True)
        dp=np.array([1.]);np_=np.array([1.])
        for i in range(12):
            pair=t[i-1]+t[i]
            if pair[1]=='H': N=[.6*vp,.6];D=[.6*vp,1.2,1]
            else:
                coop=pair=='CC';N=[.2,.7,1 if coop else 0];D=[.2,.7+.2*(.6 if coop else h),1]
            dp=poly.polymul(dp,D);np_=poly.polymul(np_,N)
        cp=poly.polysub(dp,np_); roots=poly.polyroots(cp)
        eig=np.linalg.eigvals(A);ii,jj=linear_sum_assignment(abs(eig[:,None]-roots[None,:]))
        y=np.zeros(24);y[0]=.1;y[1]=-.1
        test=np.array([.1+.2j,.2+.5j,1+1j,2+3j]); lead=cp[-1]
        residual=max(abs(np.linalg.det(z*np.eye(24)-A)-poly.polyval(z,cp)/lead)/max(1e-100,abs(np.linalg.det(z*np.eye(24)-A))) for z in test)
        checks.append(dict(vp=vp,hac=h,characteristic_relative_residual=float(residual),abscissa=spectral_abscissa(t,vp,h),norm80=float(np.linalg.norm(expm(A*80)@y)),norm400=float(np.linalg.norm(expm(A*400)@y)),norm0=float(np.linalg.norm(y))))
    save('ring_checks',checks)
    # Conditional and binomial composition. Use cached per-k,b responses.
    rng=np.random.default_rng(26095185); random=[]
    for n,p,vp,h,mode in itertools.product([8,12,20],[.25,.5,.75],[1.2,1.9],[.8,1.6,2.],['fixed','binomial']):
        cache={}; bs=[];reg=[]; ks=[]
        for j in range(500):
            k=round(n*p) if mode=='fixed' else int(rng.binomial(n,p));ks.append(k)
            if k in [0,n]:bs.append(0);reg.append(0.);continue
            idx=rng.choice(n,k,replace=False);tt=np.zeros(n,dtype=bool);tt[idx]=1;b=int(sum(tt & ~np.roll(tt,1)));bs.append(b)
            if k not in cache:
                v=[float(max(abs(product(W,ring_types(n,k,z),vp,h,True)))) for z in range(1,min(k,n-k)+1)];cache[k]=v
            v=cache[k];reg.append(v[b-1]/min(v)-1)
        random.append(dict(n=n,p=p,vp=vp,hac=h,mode=mode,samples=500,mean_k=float(np.mean(ks)),mean_b=float(np.mean(bs)),expected_b=(round(n*p)*(n-round(n*p))/(n-1) if mode=='fixed' else n*p*(1-p)),mean_regret=float(np.mean(reg)),q95_regret=float(np.quantile(reg,.95)),mc_se_b=float(np.std(bs,ddof=1)/np.sqrt(500))))
    save('random_composition',random)
    # Full internal transfer example: same product, different prefix maxima.
    internal=[]
    for t in ['HCCCHHHH','HHHCCCHH']:
        g=links(W,1.2,.8,beta=.9); prefix=np.ones(len(W),complex);allp=[]
        for i in range(1,len(t)):
            prefix*=g[t[i-1]+t[i]];allp.append(float(max(abs(prefix))))
        maxlink=max(float(max(abs(g[t[i-1]+t[i]]))) for i in range(1,len(t)))
        internal.append(dict(types=t,tail=allp[-1],max_prefix=max(allp),max_link=maxlink))
    save('internal',internal)
    print('Frequency studies complete',flush=True)

if __name__=='__main__':main()
