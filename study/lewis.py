"""(a) per-fold CV breakdown  (b) corr(|r|,1/tau) for Danckwerts  (c) SEPARATE THERMAL DISPERSION.

(c): tanks-in-series locks Pe_mass = Pe_heat. Add one parameter lam in [0,1) that applies
extra AXIAL SMOOTHING TO TEMPERATURE ONLY after each tank:
      T_i <- (1-lam) T_i + lam T_{i-1}
lam=0 -> current model (Le=1).  lam>0 -> Pe_heat < Pe_mass  (heat disperses more than mass).
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
R=8.314; Tref=430.0; n=35
tr=pd.read_csv("train_dataset.csv"); y=tr.overall_yield.values
F=tr.flow_rate_L_min.values; Ti=tr.inlet_temperature_K.values
L=tr.length_m.values; Tj=tr.jacket_temperature_K.values; C=tr.concentration_mol_L.values
KF=KFold(5,shuffle=True,random_state=0); INC=np.load("p_final_disp.npy")

def tis(q,F,Ti,L,Tj,C,lam=0.0,iters=12):
    lk1,E1,lk2,E2,lh,q1,q2=q[:7]; h=np.exp(lh); theta=(L/F)/n
    a=np.ones(len(F)); b=np.zeros(len(F)); T=Ti.copy(); Tprev=Ti.copy()
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
        Tnew=(1-lam)*Tk+lam*Tprev          # <-- extra thermal dispersion only
        Tprev=Tk; T=Tnew
    return 100*b

LO=np.array([-15,0,-15,0,-10,-60,-60.]); HI=np.array([15,800,15,800,10,60,60.])
def fit(idx,lam,s=0):
    f,t,l,j,c,yy=F[idx],Ti[idx],L[idx],Tj[idx],C[idx],y[idx]
    rng=np.random.default_rng(s); best=None
    for p0 in [INC,np.clip(INC*rng.uniform(.9,1.1,7),LO,HI)]:
        try:
            r=least_squares(lambda q:tis(q,f,t,l,j,c,lam)-yy,p0,bounds=(LO,HI),max_nfev=300)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best.x

# ---- (a) per-fold CV + (b) Danckwerts proxy ----
oof=np.zeros(len(y)); fold_rmse=[]
for k,(A,B) in enumerate(KF.split(F)):
    p=fit(A,0.0,100+k); oof[B]=tis(p,F[B],Ti[B],L[B],Tj[B],C[B],0.0)
    fold_rmse.append(np.sqrt(np.mean((oof[B]-y[B])**2)))
print("(a) per-fold CV RMSE:", np.round(fold_rmse,3), " overall %.4f"%np.sqrt(np.mean((np.clip(oof,0,100)-y)**2)))
pf=fit(np.arange(len(y)),0.0,7); e=y-tis(pf,F,Ti,L,Tj,C,0.0); tau=L/F
print("(b) corr(|resid|, 1/tau) = %+.3f   spearman %+.3f   -> Danckwerts inlet effect"
      %(np.corrcoef(np.abs(e),1/tau)[0,1],spearmanr(np.abs(e),1/tau).statistic))

# ---- (c) separate thermal dispersion ----
print("\n(c) SEPARATE THERMAL DISPERSION  (lam=0 is the current Le=1 model)")
print("%8s %11s %11s"%("lam","in-sample","CV RMSE"))
for lam in [0.0,0.10,0.20,0.35,0.50]:
    pl=fit(np.arange(len(y)),lam,11)
    ins=np.sqrt(np.mean((y-tis(pl,F,Ti,L,Tj,C,lam))**2))
    o=np.zeros(len(y))
    for k,(A,B) in enumerate(KF.split(F)):
        p=fit(A,lam,200+k); o[B]=tis(p,F[B],Ti[B],L[B],Tj[B],C[B],lam)
    print("%8.2f %11.4f %11.4f"%(lam,ins,np.sqrt(np.mean((np.clip(o,0,100)-y)**2))),flush=True)
