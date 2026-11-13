from pathlib import Path
import csv,json,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from model import *
from nonlinear_studies import ARR
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reproducibility/results';O=ROOT/'manuscript/figs';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
COL=['#31588a','#369fb5','#15946c','#d64a38']
def rd(n):return list(csv.DictReader((R/(n+'.csv')).open()))
def save(fig,n):fig.savefig(O/('fig'+str(n)+'.pdf'),bbox_inches='tight');plt.close(fig)
W=np.geomspace(.1,10,1800)
fig,axs=plt.subplots(2,1,figsize=(3.5,3.9),layout='constrained')
for ax,par in zip(axs,[[(1.2,.8),(1.2,2.)],[(1.9,1.6)]]):
    for vp,h in par:
        vals=[refined_peak(lambda w:product(w,ring_types(12,4,b),vp,h,True))[0] for b in range(1,5)]
        ax.plot(range(1,5),vals,'o-',label=f"V'={vp}, h={h} s",markersize=3)
        b=int(np.argmin(vals));ax.plot(b+1,vals[b],'o',ms=6,color=COL[b])
    ax.axhline(1,color='gray',ls=':',lw=.6);ax.set(xticks=range(1,5),ylim=(.8,1.65),ylabel=r'$\Theta$');ax.legend()
axs[0].set_title('(a) Endpoint regimes',loc='left');axs[1].set_title('(b) Interior propagation optimum',loc='left');axs[1].set_xlabel('CAV runs b');save(fig,2)
grid=rd('ring_grid');fig,axs=plt.subplots(1,2,figsize=(7.16,2.65),layout='constrained')
for ax,key,title in zip(axs,['best_b','stable_best_b'],['(a) Unconstrained propagation','(b) Modal-feasible propagation']):
    z=np.array([int(x[key]) for x in grid]).reshape(37,57)
    ax.imshow(z,origin='lower',aspect='auto',extent=[.175,3.025,.175,2.025],cmap=ListedColormap(['#eeeeee']+COL),vmin=0,vmax=4,interpolation='nearest')
    ax.axvspan(.175,.6,facecolor='gray',alpha=.25,hatch='//');ax.axhline(.9,color='black',ls=':',lw=.7)
    ax.set(xlabel=r'$h_{ac}$ (s)',ylabel=r"$V'(s_H^*)$ (s$^{-1}$)",title=title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL[i],label=f'b={i+1}') for i in range(4)]+([Patch(color='#eeeeee',label='none stable')] if key=='stable_best_b' else []),loc='upper right',ncol=2)
save(fig,3)
fig,axs=plt.subplots(2,1,figsize=(3.5,4.1),layout='constrained')
hs=np.linspace(.4,2.6,160);hh,cc,hc=[links(W,1.2,1.5,beta=.9)[k] for k in ['HH','CC','HC']]
head=refined_peak(lambda w:product(w,'CCCHHHHH',1.2,1.5,beta=.9))[0]
other=[refined_peak(lambda w:product(w,'HCCCHHHH',1.2,h,beta=.9))[0] for h in hs]
axs[0].plot(hs,np.full(len(hs),head),label='head');axs[0].plot(hs,other,label='non-head');axs[0].axvline(1.216990566,ls=':',color='k',lw=.8);axs[0].set(xlabel=r'$h_{ac}$ (s)',ylabel=r'$\Theta$',title='(a) Nominal endpoint comparison');axs[0].legend()
for t,lab in [('CCCHHHHH','head'),('HCCCHHHH','non-head')]:axs[1].semilogx(W,abs(product(W,t,.8,2.,beta=.9,kv=.4)),label=lab)
axs[1].set(xlabel=r'$\omega$ (rad/s)',ylabel='magnitude',title='(b) Converse counterexample',xlim=(.1,10),ylim=(0,1));axs[1].legend(loc='lower left')
ins=axs[1].inset_axes([.48,.47,.48,.45]); ww=np.linspace(.1,.14,150)
for t,lab in [('CCCHHHHH','head'),('HCCCHHHH','other')]:ins.plot(ww,abs(product(ww,t,.8,2.,beta=.9,kv=.4)),label=lab)
ins.tick_params(labelsize=5);ins.set_title('low-frequency peak detail',fontsize=5);save(fig,4)
top=rd('topologies');fig,ax=plt.subplots(figsize=(3.5,2.5),layout='constrained')
for typ,ls,col in [('HCCCHHHH','-',COL[0]),('HHHCCCHH','--',COL[3])]:
    for topo,marker in [('PF','o'),('PLF','s'),('2PF','^'),('MPF','D'),('BD','x')]:
        rr=[x for x in top if x['types']==typ and x['topology']==topo and x['normalized']=='1'];rr.sort(key=lambda x:float(x['gain']))
        ax.plot([float(x['gain']) for x in rr],[float(x['theta']) for x in rr],ls,marker=marker,ms=3,color=col,label=topo if typ=='HCCCHHHH' else None)
ax.set(xlabel='total nonlocal gain g (s$^{-1}$)',ylabel='head-to-tail band peak');ax.legend(ncol=3);save(fig,5)
non=rd('nonlinear');fig,axs=plt.subplots(2,1,figsize=(3.5,3.7),layout='constrained')
for ax,vp in zip(axs,[1.2,1.9]):
    for h,col in [(.8,COL[0]),(1.6,COL[1]),(2.,COL[3])]:
        for physical,ls in [('False','-'),('True','--')]:
            rr=sorted([x for x in non if float(x['vp'])==vp and float(x['hac'])==h and x['physical']==physical and float(x['amplitude'])==2],key=lambda x:int(x['b']))
            e=np.array([float(x['energy']) for x in rr]);ax.plot(range(1,5),e/max(e),ls,marker='o',ms=3,color=col,label=f'h={h} s' if physical=='False' else None)
    ax.set(xticks=range(1,5),ylabel='normalized tail energy',title=f"V'={vp}",ylim=(0,1.08));ax.legend(ncol=3)
axs[-1].set_xlabel('CAV runs b');save(fig,6)
print('Figures 2--6 generated')
