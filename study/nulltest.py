"""TWO CONTROL TESTS for the 'missing dispersion' hypothesis.

TEST 1 - SYNTHETIC NULL CONTROL
  Generate yields from the fitted 7-param model at the SAME 150 design points.
  That world contains ZERO dispersion by construction.
  Refit the SAME model with the SAME pipeline, inspect residual-vs-predicted-yield.
    hump present  -> artifact of boundedness / finite-N / optimizer
    hump absent   -> compression confound ruled out, dispersion survives

TEST 2 - RESIDUAL vs GRADIENT |dY/dtau|
  Dispersion smooths concentration gradients, so its signature should track
  gradient magnitude. Gradient is NOT bounded by 0/100, so this sidesteps the
  compression confound entirely.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import least_squares

R=8.314; Tref=430.0; N=300
tr=pd.read_csv('train_dataset.csv'); y=tr.overall_yield.values
F=tr.flow_rate_L_min.values; Ti=tr.inlet_temperature_K.values
L=tr.length_m.values; Tj=tr.jacket_temperature_K.values; C=tr.concentration_mol_L.values
p_true=np.load('p300.npy')

def sim(q,F,Ti,L,Tj,C):
    lk1,E1,lk2,E2,lh,q1,q2=q; h=np.exp(lh)
    d=L/F/N; a=np.ones(len(F)); b=np.zeros(len(F)); T=Ti.copy()
    def dv(a,b,T):
        a=np.clip(a,0,None); b=np.clip(b,0,None); T=np.clip(T,150,1500)
        inv=1/T-1/Tref
        r1=np.exp(lk1-E1*1e3/R*inv)*a; r2=np.exp(lk2-E2*1e3/R*inv)*b
        return -r1, r1-r2, h*(Tj-T)+C*(q1*r1+q2*r2)
    for i in range(N):
        A1,B1,T1=dv(a,b,T); A2,B2,T2=dv(a+.5*d*A1,b+.5*d*B1,T+.5*d*T1)
        A3,B3,T3=dv(a+.5*d*A2,b+.5*d*B2,T+.5*d*T2); A4,B4,T4=dv(a+d*A3,b+d*B3,T+d*T3)
        a=np.clip(a+d/6*(A1+2*A2+2*A3+A4),0,None); b=np.clip(b+d/6*(B1+2*B2+2*B3+B4),0,None)
        T=np.clip(T+d/6*(T1+2*T2+2*T3+T4),150,1500)
    return 100*b

LO=np.array([-15,0,-15,0,-10,-60,-60.]); HI=np.array([15,800,15,800,10,60,60.])
def fit(target,ns=14,seed=11):
    rng=np.random.default_rng(seed); best=None
    st=[p_true]+[np.clip(p_true*rng.uniform(.85,1.15,7),LO,HI) for _ in range(ns)]
    for p0 in st:
        try:
            r=least_squares(lambda q:sim(q,F,Ti,L,Tj,C)-target,p0,bounds=(LO,HI),max_nfev=400)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best.x

BINS=[(0,1),(1,10),(10,30),(30,50),(50,70),(70,90),(90,101)]
def table(pred,e,label):
    print('\n%s'%label)
    print('%12s %5s %9s %9s'%('pred bin','n','RMSE','mean res'))
    for lo,hi in BINS:
        m=(pred>=lo)&(pred<hi)
        if m.sum()>2:
            print('%6.0f-%-5.0f %5d %9.2f %9.2f'%(lo,hi,m.sum(),np.sqrt(np.mean(e[m]**2)),e[m].mean()))
    print('  corr(|res|,pred) = %.3f'%np.corrcoef(np.abs(e),pred)[0,1])

# ---------------- TEST 1 : synthetic null ----------------
print('='*66); print('TEST 1 - SYNTHETIC NULL CONTROL (no dispersion exists)'); print('='*66)
y_syn=sim(p_true,F,Ti,L,Tj,C)              # ground truth of a dispersion-free world
p_syn=fit(y_syn)
pred_syn=sim(p_syn,F,Ti,L,Tj,C); e_syn=y_syn-pred_syn
print('recovered params vs true:'); print('  true',np.round(p_true,3)); print('  refit',np.round(p_syn,3))
print('synthetic refit RMSE %.4f  (should be ~0 if pipeline is unbiased)'%np.sqrt(np.mean(e_syn**2)))
table(pred_syn,e_syn,'SYNTHETIC (null world) residual vs predicted yield')

# real data for comparison
pred_real=sim(p_true,F,Ti,L,Tj,C); e_real=y-pred_real
table(pred_real,e_real,'REAL DATA residual vs predicted yield')

# ---------------- TEST 2 : residual vs gradient ----------------
print('\n'+'='*66); print('TEST 2 - RESIDUAL vs |dY/dtau|  (not bounded by 0/100)'); print('='*66)
eps=0.02
Yp=sim(p_true,F,Ti,L*(1+eps),Tj,C); Ym=sim(p_true,F,Ti,L*(1-eps),Tj,C)
tau=L/F
grad=np.abs((Yp-Ym)/(2*eps*tau))           # |dY/dtau|
q=np.quantile(grad,[0,.2,.4,.6,.8,1.0])
print('%22s %5s %9s %9s'%('|dY/dtau| bin','n','RMSE','mean res'))
for i in range(5):
    m=(grad>=q[i])&((grad<=q[i+1]) if i==4 else (grad<q[i+1]))
    if m.sum()>2:
        print('%10.2f-%-10.2f %5d %9.2f %9.2f'%(q[i],q[i+1],m.sum(),np.sqrt(np.mean(e_real[m]**2)),e_real[m].mean()))
print('  corr(|res|, |dY/dtau|) = %.3f'%np.corrcoef(np.abs(e_real),grad)[0,1])
print('  spearman              = %.3f'%pd.Series(np.abs(e_real)).corr(pd.Series(grad),method='spearman'))
# same check on the synthetic null, so gradient test has its own control
print('\n  [null-world control] corr(|res_syn|,|dY/dtau|) = %.3f'%np.corrcoef(np.abs(e_syn),grad)[0,1])
