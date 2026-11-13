from pathlib import Path
import itertools,json
import numpy as np
from scipy.integrate import solve_ivp
from model import *
from nonlinear_studies import simulate,save,ARR
OUT=Path(__file__).parent/'results'
rows=[]
for c,vp,h in itertools.product([.2,.5,1,2,5],[.6,1.2,1.6],np.arange(.2,3.001,.02)):
    vals=[max(abs(product(np.geomspace(.1,10,1800),ring_types(12,4,b),vp,h,True,c=c))) for b in range(1,5)]
    rows.append(dict(c=c,vp=vp,hac=float(h),best_b=int(np.argmin(vals)+1)))
save('heterogeneous_links',rows)
rows=[]
for n,k,vp,h in itertools.product([8,12,16,20],[2,3,4,6],[.6,1.2,1.9],np.arange(.6,2.201,.05)):
    if not 0<k<n:continue
    vals=[max(abs(product(np.geomspace(.1,10,1800),ring_types(n,k,b),vp,h,True))) for b in range(1,min(k,n-k)+1)]
    rows.append(dict(n=n,k=k,vp=vp,hac=float(h),best_b=int(np.argmin(vals)+1),max_b=min(k,n-k)))
save('size_sensitivity',rows)
# Necessary numerical validation: product vs independently assembled state response.
w=np.geomspace(.01,10,31);errors=[]
for t in ['CCCHHHHH','HCCCHHHH','HHHCCCHH']:
    r,_=response(t,w,beta=.9)
    p=product(w,t,1.2,1.5,beta=.9)
    errors.append(float(max(abs(r-p))))
assert max(errors)<1e-11
# Small-signal nonlinear vs independent linear integration, physical spacings.
cfg=[dict(types=ARR[1],vp=1.2,hac=2.,amplitude=1e-3,physical=True)]
rr=simulate(cfg,dt=.002,T=80)[0]
A,B0,B1=matrices(ARR[1],1.2,2.)
def fun(t,y):
    e=np.exp(-t/6);u=.001*e*(np.sin(.9*t)+.5*np.sin(4*t))
    du=.001*e*(.9*np.cos(.9*t)-np.sin(.9*t)/6+2*np.cos(4*t)-np.sin(4*t)/12)
    return A@y+B0*u+B1*du
tt=np.arange(0,80.0001,.002);sol=solve_ivp(fun,(0,80),np.zeros(len(A)),method='DOP853',t_eval=tt,rtol=1e-10,atol=1e-13)
energy=np.trapezoid(sol.y[-1]**2,tt)
rel=abs(rr['energy']/energy-1);assert rel<.005
zero=simulate([dict(types=ARR[1],vp=1.2,hac=2.,amplitude=0,physical=True)],T=1)[0]
assert zero['energy']<1e-20
checks=dict(product_matrix_max_error=max(errors),small_signal_relative_energy_error=rel,zero_input_energy=zero['energy'])
(OUT/'validation_checks.json').write_text(json.dumps(checks,indent=2))
print(checks,flush=True)
# Physical safety comparisons across penetration rates, speeds, actuation limits.
rows=[]
for k,vp,h,b,sat in itertools.product([3,6,9],[.6,1.2,1.9],[.8,1.6,2.],[1,3],[False,True]):
    t=ring_types(12,k,b);j=t.index('H');t=t[j:]+t[:j]
    rows.append(dict(types=t,k=k,vp=vp,hac=h,b=b,amplitude=4.,physical=True,saturation=sat))
if not (OUT/'penetration_safety.csv').exists():save('penetration_safety',simulate(rows))
print('Additional studies complete',flush=True)
