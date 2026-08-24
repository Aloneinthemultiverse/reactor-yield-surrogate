"""Diagnose fold 0 (CV RMSE 14.1 vs ~2-4 elsewhere) and test whether ensembling fixes it."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
R=8.314; Tref=430.0; n=35
tr=pd.read_csv("train_dataset.csv"); y=tr.overall_yield.values
F=tr.flow_rate_L_min.values; Ti=tr.inlet_temperature_K.values
L=tr.length_m.values; Tj=tr.jacket_temperature_K.values; C=tr.concentration_mol_L.values
KF=KFold(5,shuffle=True,random_state=0); INC=np.load("p_final_disp.npy")
LO=np.array([-15,0,-15,0,-10,-60,-60.]); HI=np.array([15,800,15,800,10,60,60.])

def tis(q,F,Ti,L,Tj,C,lam=0.0,iters=12):
    lk1,E1,lk2,E2,lh,q1,q2=q; h=np.exp(lh); theta=(L/F)/n
    a=np.ones(len(F)); b=np.zeros(len(F)); T=Ti.copy(); Tp=Ti.copy()
    for _ in range(n):
        a_in,b_in,T_in=a,b,T; Tk=T_in.copy()
        for _ in range(iters):
            Tk=np.clip(Tk,150,1500); inv=1/Tk-1/Tref
            k1=np.exp(np.clip(lk1-E1*1e3/R*inv,-50,50)); k2=np.exp(np.clip(lk2-E2*1e3/R*inv,-50,50))
            an=a_in/(1+theta*k1); bn=(b_in+theta*k1*an)/(1+theta*k2)
            Tn=np.clip((T_in+theta*h*Tj+theta*C*(q1*k1*an+q2*k2*bn))/(1+theta*h),150,1500)
            if np.max(np.abs(Tn-Tk))<1e-8: Tk=Tn; break
            Tk=0.5*Tk+0.5*Tn
        inv=1/Tk-1/Tref
        k1=np.exp(np.clip(lk1-E1*1e3/R*inv,-50,50)); k2=np.exp(np.clip(lk2-E2*1e3/R*inv,-50,50))
        a=np.clip(a_in/(1+theta*k1),0,None); b=np.clip((b_in+theta*k1*a)/(1+theta*k2),0,None)
        T=(1-lam)*Tk+lam*Tp; Tp=Tk
    return 100*b

def fit(idx,lam=0.0,s=0,ns=2):
    f,t,l,j,c,yy=F[idx],Ti[idx],L[idx],Tj[idx],C[idx],y[idx]
    rng=np.random.default_rng(s); best=None
    st=[INC]+[np.clip(INC*rng.uniform(.9,1.1,7),LO,HI) for _ in range(ns)]
    for p0 in st:
        try:
            r=least_squares(lambda q:tis(q,f,t,l,j,c,lam)-yy,p0,bounds=(LO,HI),max_nfev=300)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best.x

folds=list(KF.split(F))
A0,B0=folds[0]
p0=fit(A0,0.0,100)
pred0=tis(p0,F[B0],Ti[B0],L[B0],Tj[B0],C[B0]); e0=y[B0]-pred0
print("FOLD 0: %d held-out rows, RMSE %.3f"%(len(B0),np.sqrt(np.mean(e0**2))))
o=np.argsort(-np.abs(e0))
print("\n%6s %8s %8s %8s | %6s %5s %7s %6s %7s %7s"%("row","actual","pred","err","F","C","Ti","L","Tj","T_avg"))
for i in o[:8]:
    g=B0[i]; print("%6d %8.2f %8.2f %8.2f | %6.1f %5.2f %7.1f %6.2f %7.1f %7.1f"
        %(g,y[g],pred0[i],e0[i],F[g],C[g],Ti[g],L[g],Tj[g],(Ti[g]+Tj[g])/2))
cum=np.cumsum(np.sort(e0**2)[::-1]); tot=cum[-1]
print("\ntop 1 row = %.0f%% of fold-0 squared error;  top 3 = %.0f%%;  top 5 = %.0f%%"
      %(100*cum[0]/tot,100*cum[2]/tot,100*cum[4]/tot))

print("\nis fold 0's PARAMETER FIT an outlier?")
P=np.array([fit(A,0.0,100+k) for k,(A,_) in enumerate(folds)])
for i,nm in enumerate(["lk1","Ea1","lk2","Ea2","lh","q1","q2"]):
    z=(P[0,i]-P[1:,i].mean())/(P[1:,i].std()+1e-9)
    print("  %5s fold0=%9.3f  others %9.3f +/- %-8.3f  z=%+.2f"%(nm,P[0,i],P[1:,i].mean(),P[1:,i].std(),z))

print("\nDOES ENSEMBLING / lam FIX FOLD 0?")
def foldwise(lam,boot):
    r=[]
    for k,(A,B) in enumerate(folds):
        if boot:
            ps=[fit(A[np.random.default_rng(5000+10*k+i).integers(0,len(A),len(A))],lam,600+i,1) for i in range(8)]
            pr=np.mean([tis(p,F[B],Ti[B],L[B],Tj[B],C[B],lam) for p in ps],0)
        else:
            pr=tis(fit(A,lam,100+k),F[B],Ti[B],L[B],Tj[B],C[B],lam)
        r.append(np.sqrt(np.mean((np.clip(pr,0,100)-y[B])**2)))
    return np.array(r)
for lam,boot,tag in [(0.0,False,"single      lam=0"),(0.0,True,"bootstrap   lam=0"),
                     (0.10,False,"single      lam=0.10"),(0.10,True,"bootstrap   lam=0.10")]:
    r=foldwise(lam,boot)
    print("  %-22s folds %s  overall %.4f"%(tag,np.round(r,2),np.sqrt(np.mean(r**2))),flush=True)
