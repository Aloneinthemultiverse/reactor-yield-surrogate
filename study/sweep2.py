"""Remaining physics structures, optimised: only ACTIVE parameters go to least_squares
(pinned ones were previously costing Jacobian columns for nothing)."""
import pandas as pd, numpy as np, warnings, sys; warnings.filterwarnings('ignore')
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

R=8.314; Tref=430.0; N=60
tr=pd.read_csv('train_dataset.csv'); y=tr.overall_yield.values
F=tr.flow_rate_L_min.values; Ti=tr.inlet_temperature_K.values
L=tr.length_m.values; Tj=tr.jacket_temperature_K.values; C=tr.concentration_mol_L.values
KF=KFold(5,shuffle=True,random_state=0)
FULL=np.array([2.4933,42.5987,-0.4621,206.0158,1.2134, 0.,0., -30.,60., 1.,1.])

def sim(p,F,Ti,L,Tj,C):
    lk1,E1,lk2,E2,lh,q1,q2,lk3,E3,n1,n2=p; h=np.exp(lh)
    d=L/F/N; a=np.ones(len(F)); b=np.zeros(len(F)); T=Ti.copy()
    def dv(a,b,T):
        a=np.clip(a,1e-12,None); b=np.clip(b,1e-12,None); T=np.clip(T,150,1500)
        inv=1/T-1/Tref
        r1=np.exp(lk1-E1*1e3/R*inv)*C**(n1-1)*a**n1
        r2=np.exp(lk2-E2*1e3/R*inv)*C**(n2-1)*b**n2
        r3=np.exp(lk3-E3*1e3/R*inv)*C**(n1-1)*a**n1
        return -(r1+r3), r1-r2, h*(Tj-T)+C*(q1*r1+q2*r2)
    for i in range(N):
        A1,B1,T1=dv(a,b,T); A2,B2,T2=dv(a+.5*d*A1,b+.5*d*B1,T+.5*d*T1)
        A3,B3,T3=dv(a+.5*d*A2,b+.5*d*B2,T+.5*d*T2); A4,B4,T4=dv(a+d*A3,b+d*B3,T+d*T3)
        a=np.clip(a+d/6*(A1+2*A2+2*A3+A4),0,None)
        b=np.clip(b+d/6*(B1+2*B2+2*B3+B4),0,None)
        T=np.clip(T+d/6*(T1+2*T2+2*T3+T4),150,1500)
    return 100*b

LO=np.array([-15,0,-15,0,-10,-60,-60,-35,0,0.3,0.3],float)
HI=np.array([ 15,800,15,800, 10, 60, 60, 15,800,3.0,3.0],float)

def active(heat,par,ord_):
    A=[0,1,2,3,4]
    if heat: A+=[5,6]
    if par:  A+=[7,8]
    if ord_: A+=[9,10]
    return np.array(A)

def fit(idx,heat,par,ord_,nstart,seed):
    f,t,l,j,c,yy=F[idx],Ti[idx],L[idx],Tj[idx],C[idx],y[idx]
    A=active(heat,par,ord_); rng=np.random.default_rng(seed); best=None; bx=None
    def expand(q):
        p=FULL.copy(); p[A]=q; return p
    starts=[FULL[A]]
    for _ in range(nstart):
        s=FULL.copy(); s[:5]*=rng.uniform(.85,1.15,5)
        if heat: s[5],s[6]=rng.uniform(-15,15,2)
        if par:  s[7],s[8]=rng.uniform(-6,2),rng.uniform(30,260)
        if ord_: s[9],s[10]=rng.uniform(.7,1.5,2)
        starts.append(s[A])
    for q0 in starts:
        try:
            r=least_squares(lambda q:sim(expand(q),f,t,l,j,c)-yy,np.clip(q0,LO[A],HI[A]),
                            bounds=(LO[A],HI[A]),max_nfev=200)
            if best is None or r.cost<best.cost: best=r; bx=expand(r.x)
        except Exception: pass
    return bx if bx is not None else FULL

def evaluate(heat,par,ord_,tag):
    pf=fit(np.arange(len(y)),heat,par,ord_,5,99)
    ins=np.sqrt(np.mean((y-sim(pf,F,Ti,L,Tj,C))**2))
    oof=np.zeros(len(y))
    for k,(A,B) in enumerate(KF.split(F)):
        p=fit(A,heat,par,ord_,3,k); oof[B]=sim(p,F[B],Ti[B],L[B],Tj[B],C[B])
    cv=np.sqrt(np.mean((oof-y)**2)); r2=1-np.sum((oof-y)**2)/np.sum((y-y.mean())**2)
    print('%-26s npar=%2d in=%6.3f CV=%6.3f R2=%.4f'%(tag,len(active(heat,par,ord_)),ins,cv,r2),flush=True)
    if heat: print('     q1=%.3f q2=%.3f'%(pf[5],pf[6]),flush=True)
    if par:  print('     Ea3=%.1f k3ref=%.4f'%(pf[8],np.exp(pf[7])),flush=True)
    if ord_: print('     n1=%.3f n2=%.3f'%(pf[9],pf[10]),flush=True)
    return dict(tag=tag,p=pf,oof=oof,cv=cv,flags=(heat,par,ord_))

out=[]
for h,p_,o,tag in [(1,0,0,'HEAT'),(0,1,0,'PARALLEL'),(1,0,1,'HEAT+ORDERS'),
                   (0,1,1,'PARALLEL+ORDERS'),(1,1,0,'HEAT+PARALLEL'),(1,1,1,'HEAT+PAR+ORDERS')]:
    out.append(evaluate(h,p_,o,tag))
b=min(out,key=lambda r:r['cv'])
print('\nBEST OF THESE: %s CV=%.4f'%(b['tag'],b['cv']),flush=True)
print('(reference: series+ORDERS = 9.64, series 1st-order = 9.75)',flush=True)
np.save('sweep2_best.npy',b['p']); np.save('sweep2_flags.npy',np.array(b['flags']))
np.save('sweep2_oof.npy',b['oof'])
