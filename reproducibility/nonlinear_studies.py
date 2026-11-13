"""Vectorized, stage-correct RK4 with prescribed external leader.
No virtual leader spacing is integrated. Packet reception is a sampled process;
received feedforward is held between 0.1-s samples. Communication runs use .01 s.
"""
from pathlib import Path
import csv,json,itertools,time
import numpy as np
OUT=Path(__file__).parent/'results'
ARR={1:'HCCCCHHHHHHH',2:'HCCHHHCCHHHH',3:'HCCHHCHHCHHH',4:'HCHHCHHCHHCH'}
def save(name,rows):
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(name,len(rows),flush=True)

def simulate(configs,dt=.002,T=80,communication=False,trace=False):
    B=len(configs); n=len(configs[0]['types']);m=n-1
    typ=np.array([[c=='C' for c in r['types']] for r in configs]); cav=typ[:,1:]
    coop=cav&typ[:,:-1]
    def ar(k,default):return np.array([r.get(k,default) for r in configs])[:,None]
    vp=ar('vp',1.2);hac=ar('hac',2.);beta=ar('beta',.6); amp=ar('amplitude',2.)
    V0=ar('V0',32.);L=ar('L',8.);ell=ar('ell',5.)
    ss=ell-L*np.log(vp*L/V0);vs=V0-vp*L
    h=np.where(coop,.6,hac);ka=coop.astype(float)
    physical=ar('physical',False).astype(bool)
    seq=np.where(cav & physical,5+h*vs,ss)
    s=seq.copy();v=np.broadcast_to(vs,(B,m)).copy()
    E=np.zeros(B); minspace=np.full(B,np.inf); minttc=np.full(B,np.inf);amax=np.zeros(B);amin=np.zeros(B);aenergy=np.zeros(B);distance=np.zeros(B);endenergy=np.zeros(B)
    # Noise frequency range and amplitude are explicit. RMS is normalized on [0,T].
    freq=np.linspace(.3,5,16); phases=np.array([np.random.default_rng(r.get('seed',0)).uniform(0,2*np.pi,16) for r in configs])
    noise=ar('noise',False).astype(bool);rms=ar('rms',.5)
    ts=np.linspace(0,T,4001); normalization=np.sqrt(np.mean((np.sin(ts[:,None,None]*freq+phases[None,:,:]).sum(axis=2))**2,axis=0))[:,None] if np.any(noise) else np.ones((B,1))
    def lead(t):
        e=np.exp(-t/6); u=amp*e*(np.sin(.9*t)+.5*np.sin(4*t));du=amp*e*(.9*np.cos(.9*t)-np.sin(.9*t)/6+2*np.cos(4*t)-np.sin(4*t)/12)
        if np.any(noise):
            u=np.where(noise,rms*np.sin(freq*t+phases).sum(axis=1)[:,None]/normalization,u)
            du=np.where(noise,rms*(freq*np.cos(freq*t+phases)).sum(axis=1)[:,None]/normalization,du)
        return u,du
    lossp=ar('loss',0);burst=ar('burst',0);delta=ar('delta',0)
    fallback=ar('fallback',False).astype(bool)
    brake=ar('brake',-3)[:,0]; accelerate=ar('accelerate',2)[:,0]
    group_keys=[(r.get('seed',0),r.get('loss',0),r.get('burst',0),r.get('delta',0)) for r in configs]
    unique=list(dict.fromkeys(group_keys)); group=np.array([unique.index(k) for k in group_keys])
    bad=np.zeros((B,m),bool);held=np.zeros((B,m)); rng=np.random.default_rng(197704)
    history=[]; histstep=max(1,round(.1/dt)); hs=h.copy(); sat=ar('saturation',False).astype(bool)
    def deriv(t,sp,vel):
        u,du=lead(t);pred=np.concatenate((vs+u,vel[:,:-1]),axis=1)
        hh=hs if communication else h
        raw=np.where(cav,.2*(sp-seq-hh*(vel-vs))+.7*(pred-vel),.6*(V0*(1-np.exp(np.clip(-(sp-ell)/L,-50,50)))-vel)+beta*(pred-vel))
        # For physical fallback use fixed s0=5, not a mode-dependent bias.
        if communication and np.any(physical):raw=np.where(cav&physical,.2*(sp-5-hh*vel)+.7*(pred-vel),raw)
        acc=np.empty_like(raw)
        for i in range(m):
            pa=du[:,0] if i==0 else acc[:,i-1]
            acc[:,i]=raw[:,i]+(ka[:,i]*held[:,i] if communication else ka[:,i]*pa)
            acc[:,i]=np.where(sat[:,0],np.clip(acc[:,i],brake,accelerate),acc[:,i])
        return pred-vel,acc
    traces=[]
    for j in range(round(T/dt)):
        t=j*dt
        if communication and j%histstep==0:
            # Current no-feedforward acceleration estimate is updated with received history.
            _,a_now=deriv(t,s,v);_,du=lead(t); pa=np.concatenate((du,a_now[:,:-1]),axis=1);history.append(pa.copy())
            u1=rng.random((len(unique),m))[group];u2=rng.random((len(unique),m))[group]
            iid=u1<lossp
            leave=np.divide(1,burst,out=np.ones_like(burst,dtype=float),where=burst>0)
            enter=np.divide(lossp*leave,1-lossp,out=np.zeros_like(lossp),where=lossp<1)
            bb=np.where(bad,u1>=leave,u2<enter)
            bad=np.where(burst>0,bb,iid)
            lag=np.rint(delta[:,0]/.1).astype(int)
            received=np.array([history[max(0,len(history)-1-int(lag[q]))][q] if len(history)>lag[q] else np.zeros(m) for q in range(B)])
            held=np.where(bad,0,received)
            if len(history)>6:history.pop(0)
        if communication:
            target=np.where(coop&bad&fallback,hac,h)
            hs+=(target-hs)*(dt/.5)
        k1s,k1v=deriv(t,s,v);k2s,k2v=deriv(t+dt/2,s+k1s*dt/2,v+k1v*dt/2)
        k3s,k3v=deriv(t+dt/2,s+k2s*dt/2,v+k2v*dt/2);k4s,k4v=deriv(t+dt,s+k3s*dt,v+k3v*dt)
        vn=v+dt/6*(k1v+2*k2v+2*k3v+k4v);sn=s+dt/6*(k1s+2*k2s+2*k3s+k4s)
        energy=.5*dt*((v[:,-1]-vs[:,0])**2+(vn[:,-1]-vs[:,0])**2);E+=energy
        if t>=T-10:endenergy+=energy
        minspace=np.minimum(minspace,s.min(axis=1));amax=np.maximum(amax,k1v.max(axis=1));amin=np.minimum(amin,k1v.min(axis=1))
        u,_=lead(t); pred=np.concatenate((vs+u,v[:,:-1]),axis=1);closing=v-pred
        ttc=np.divide(s,closing,out=np.full_like(s,np.inf),where=closing>1e-6);minttc=np.minimum(minttc,ttc.min(axis=1))
        aenergy+=dt*np.mean(k1v*k1v,axis=1);distance+=dt*.5*(v[:,-1]+vn[:,-1])
        if trace and j%max(1,round(.02/dt))==0:traces.append(np.r_[t,v[0]-vs[0],v[-1]-vs[-1]])
        s,v=sn,vn
    rows=[]
    for q,r in enumerate(configs):
        rows.append(dict(**r,dt=dt,T=T,energy=float(E[q]),last10_energy=float(endenergy[q]),min_spacing=float(minspace[q]),min_ttc=float(minttc[q]),min_acc=float(amin[q]),max_acc=float(amax[q]),acc_rms=float(np.sqrt(aenergy[q]/T)),tail_distance=float(distance[q]),collision=int(minspace[q]<=0)))
    if trace:np.save(OUT/'tail_traces.npy',np.array(traces))
    return rows

def main():
    base=[dict(types=ARR[b],vp=vp,hac=h,b=b,physical=physical,amplitude=amp) for vp,h,b,physical,amp in itertools.product([1.2,1.9],[.8,1.6,2.],[1,2,3,4],[False,True],[.5,1.,2.])]
    if not (OUT/'nonlinear.csv').exists():save('nonlinear',simulate(base,trace=True))
    convergence=[dict(types=ARR[b],vp=vp,hac=h,b=b) for vp,h,b in itertools.product([1.2,1.9],[.8,1.6,2.],[1,4])]
    conv=[]
    for dt,T in [(.002,80),(.001,80),(.0005,80),(.002,120)]:
        key='convergence_'+str(dt)+'_'+str(T)
        if (OUT/(key+'.csv')).exists():
            conv+=list(csv.DictReader((OUT/(key+'.csv')).open()))
        else:
            rr=simulate(convergence,dt,T);save(key,rr);conv+=rr
        print('convergence',dt,T,flush=True)
    save('convergence',conv)
    pos=[dict(types='H'*start+'CCCC'+'H'*(8-start),vp=1.2,hac=2.,start=start,physical=physical) for start,physical in itertools.product(range(1,8),[False,True])]
    if not (OUT/'positions.csv').exists():save('positions',simulate(pos))
    wedge=[dict(types=ARR[b],vp=1.9,hac=float(h),b=b) for h,b in itertools.product(np.linspace(.5,1.65,11),[1,2,3,4])]
    if not (OUT/'nonlinear_wedge.csv').exists():save('nonlinear_wedge',simulate(wedge))
    ov=[dict(types=ARR[b],vp=1.2,hac=h,b=b,V0=V0,L=L,ell=ell,physical=True) for b,h,V0,L,ell in itertools.product([1,4],[.8,2.],[28.,32.,36.],[6.,8.,10.],[4.,5.,6.])]
    if not (OUT/'ov_sensitivity.csv').exists():save('ov_sensitivity',simulate(ov))
    noise=[dict(types=ARR[b],vp=vp,hac=h,b=b,noise=True,seed=seed,rms=rms) for (vp,h),b,seed,rms in itertools.product([(1.2,.8),(1.2,2.),(1.9,1.6)],[1,2,3,4],range(30),[.2,.5,1.])]
    # Group by noise seed so all arrangements receive the same realization.
    if not (OUT/'noise.csv').exists():save('noise',simulate(noise))
    comm=[dict(types=ARR[b],vp=vp,hac=h,b=b,seed=seed,loss=loss,burst=burst,delta=delta,fallback=fallback,physical=True,saturation=True) for (vp,h),b,seed,(loss,burst,delta),fallback in itertools.product([(1.2,.8),(1.2,2.),(1.9,1.6)],[1,2,3,4],range(30),[(0,0,0),(.05,0,0),(.1,0,0),(.2,0,0),(.1,3,0),(.1,5,0),(0,0,.1),(0,0,.2),(0,0,.5)],[False,True])]
    if not (OUT/'communication.csv').exists():save('communication',simulate(comm,dt=.01,communication=True))
    safety=[dict(types=ARR[b],vp=vp,hac=h,b=b,amplitude=amp,physical=True,saturation=sat,brake=brake,accelerate=accel) for vp,h,b,amp,sat,(brake,accel) in itertools.product([.6,1.2,1.9],[.8,1.6,2.],[1,2,3,4],[2.,4.],[False,True],[(-3,2),(-4,3)])]
    if not (OUT/'safety.csv').exists():save('safety',simulate(safety))
    print('Nonlinear studies complete',flush=True)
if __name__=='__main__':main()
