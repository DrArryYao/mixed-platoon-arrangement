from nonlinear_studies import *
import itertools
cases=[dict(types=ARR[b],vp=vp,hac=h,b=b,seed=seed,loss=.1,burst=burst,delta=0,fallback=fallback,physical=True,saturation=True) for (vp,h),b,seed,burst,fallback in itertools.product([(1.2,.8),(1.2,2.),(1.9,1.6)],[1,4],range(3),[0,5],[False,True])]
rows=[]
for dt in [.01,.005]:rows+=simulate(cases,dt=dt,communication=True)
save('communication_convergence',rows)
