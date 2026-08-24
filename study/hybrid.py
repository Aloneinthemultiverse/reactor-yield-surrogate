"""CHEMISTRY + PHYSICS + ML HYBRID.

Physics predicts the bulk; ML learns only the residual, using features derived from the
CHEMISTRY of the fitted model rather than from the raw columns:

  |dY/dtau|          steepness of the model's own conversion curve  (Spearman 0.737 with |resid|)
  |T_avg - 449.1|    distance from the selectivity crossover k2 = k1, derived from fitted Ea's
  Da1, Da2           Damkohler numbers k_i(T)*tau from the fitted rate constants
  Y_max(T)           theoretical ceiling (k1/k2)^(k2/(k2-k1)) - pure Levenspiel
  phys               the physics prediction itself

Leak-free: inside every fold the physics is refitted on the training rows only, the residual
is formed there, the ML model is trained on it, and both are applied to the held-out rows.
"""
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

R = 8.314; Tref = 430.0; N_TANKS = 35; TX = 449.1     # crossover k2=k1, derived in section 4
tr = pd.read_csv("train_dataset.csv"); te = pd.read_csv("test_dataset.csv")
y = tr.overall_yield.values
def cols(d): return (d.flow_rate_L_min.values, d.inlet_temperature_K.values,
                     d.length_m.values, d.jacket_temperature_K.values, d.concentration_mol_L.values)
F, Ti, L, Tj, C = cols(tr); tF, tTi, tL, tTj, tC = cols(te)
KF = KFold(5, shuffle=True, random_state=0)
INC = np.load("p_final_disp.npy")
LO = np.array([-15, 0, -15, 0, -10, -60, -60.]); HI = np.array([15, 800, 15, 800, 10, 60, 60.])

def tis(q, F, Ti, L, Tj, C, n=N_TANKS, iters=12):
    lk1, E1, lk2, E2, lh, q1, q2 = q; h = np.exp(lh); theta = (L / F) / n
    a = np.ones(len(F)); b = np.zeros(len(F)); T = Ti.copy()
    for _ in range(n):
        a_in, b_in, T_in = a, b, T; Tk = T_in.copy()
        for _ in range(iters):
            Tk = np.clip(Tk, 150, 1500); inv = 1/Tk - 1/Tref
            k1 = np.exp(np.clip(lk1 - E1*1e3/R*inv, -50, 50)); k2 = np.exp(np.clip(lk2 - E2*1e3/R*inv, -50, 50))
            an = a_in/(1+theta*k1); bn = (b_in+theta*k1*an)/(1+theta*k2)
            Tn = np.clip((T_in + theta*h*Tj + theta*C*(q1*k1*an + q2*k2*bn))/(1+theta*h), 150, 1500)
            if np.max(np.abs(Tn-Tk)) < 1e-8: Tk = Tn; break
            Tk = 0.5*Tk + 0.5*Tn
        inv = 1/Tk - 1/Tref
        k1 = np.exp(np.clip(lk1 - E1*1e3/R*inv, -50, 50)); k2 = np.exp(np.clip(lk2 - E2*1e3/R*inv, -50, 50))
        a = np.clip(a_in/(1+theta*k1), 0, None); b = np.clip((b_in+theta*k1*a)/(1+theta*k2), 0, None); T = Tk
    return 100*b

def fit(idx, seed=0):
    f, t, l, j, c, yy = F[idx], Ti[idx], L[idx], Tj[idx], C[idx], y[idx]
    rng = np.random.default_rng(seed); best = None
    for p0 in [INC, np.clip(INC*rng.uniform(.9, 1.1, 7), LO, HI)]:
        try:
            r = least_squares(lambda q: tis(q, f, t, l, j, c) - yy, p0, bounds=(LO, HI), max_nfev=250)
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    return best.x if best is not None else INC

def chem_features(p, F, Ti, L, Tj, C):
    """features derived from the FITTED CHEMISTRY, not the raw columns"""
    tau = L/F; Tav = (Ti+Tj)/2
    inv = 1/Tav - 1/Tref
    k1 = np.exp(p[0] - p[1]*1e3/R*inv); k2 = np.exp(p[2] - p[3]*1e3/R*inv)
    phys = tis(p, F, Ti, L, Tj, C)
    eps = 0.02
    grad = np.abs((tis(p, F, Ti, L*(1+eps), Tj, C) - tis(p, F, Ti, L*(1-eps), Tj, C))/(2*eps*tau))
    d = np.where(np.abs(k2-k1) < 1e-9, 1e-9, k2-k1)
    Ymax = 100*np.clip(k1/np.clip(k2, 1e-12, None), 1e-12, None)**(k2/d)
    return np.c_[phys, grad, np.abs(Tav-TX), Tav-TX, k1*tau, k2*tau, np.clip(Ymax, 0, 100),
                 np.log1p(k2/np.clip(k1, 1e-12, None)), tau, C, F]

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
t0 = time.time(); n = len(y)
MODELS = {"ExtraTrees": lambda: ExtraTreesRegressor(400, min_samples_leaf=3, random_state=0),
          "HistGB":     lambda: HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=8,
                                                              learning_rate=0.05, random_state=0),
          "Ridge":      lambda: RidgeCV(alphas=np.logspace(-3, 4, 30))}
oof_phys = np.zeros(n); oof_hyb = {k: np.zeros(n) for k in MODELS}
for k, (A, B) in enumerate(KF.split(F)):
    p = fit(A, 100+k)
    Xa = chem_features(p, F[A], Ti[A], L[A], Tj[A], C[A])
    Xb = chem_features(p, F[B], Ti[B], L[B], Tj[B], C[B])
    ph_a, ph_b = Xa[:, 0], Xb[:, 0]
    oof_phys[B] = ph_b
    res_a = y[A] - ph_a
    for nm, mk in MODELS.items():
        m = mk().fit(Xa, res_a)
        oof_hyb[nm][B] = ph_b + m.predict(Xb)
    print("  fold %d (%.0fs)" % (k, time.time()-t0), flush=True)

print("\nphysics alone                     CV %.4f" % rmse(np.clip(oof_phys, 0, 100), y))
for nm in MODELS:
    print("physics + %-12s on residual  CV %.4f" % (nm, rmse(np.clip(oof_hyb[nm], 0, 100), y)))
best_nm = min(MODELS, key=lambda k: rmse(np.clip(oof_hyb[k], 0, 100), y))
print("\nbest hybrid: %s  CV %.4f   vs bootstrap-averaged physics 6.500"
      % (best_nm, rmse(np.clip(oof_hyb[best_nm], 0, 100), y)))
