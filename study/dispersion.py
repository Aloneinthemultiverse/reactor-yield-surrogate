"""AXIAL DISPERSION via tanks-in-series (TIS).

A closed-closed axially-dispersed PFR is well approximated by n equal CSTRs in series:
        Pe  ~  2 (n - 1)
n -> inf recovers ideal plug flow (our current model), so the PFR is nested inside this
family and the comparison is a strict generalisation test.

Each tank, residence time theta = tau/n, steady state:
    a = a_in / (1 + theta k1)
    b = (b_in + theta k1 a) / (1 + theta k2)
    T (1 + theta h) = T_in + theta h Tj + theta C0 (q1 k1 a + q2 k2 b)
k depends on T, so each tank is closed by fixed-point iteration (vectorised over rows).

STAGE 1: in-sample fit across an n grid  -> does finite dispersion help at all?
STAGE 2: cross-validate only the promising n values.
"""
import numpy as np, pandas as pd, warnings, sys; warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

R = 8.314; Tref = 430.0
tr = pd.read_csv("train_dataset.csv"); y = tr.overall_yield.values
F = tr.flow_rate_L_min.values; Ti = tr.inlet_temperature_K.values
L = tr.length_m.values; Tj = tr.jacket_temperature_K.values; C = tr.concentration_mol_L.values
KF = KFold(5, shuffle=True, random_state=0)
P_SEED = np.array([2.720, 43.192, 0.164, 249.409, 1.179, -11.775, 11.343])
LO = np.array([-15, 0, -15, 0, -10, -60, -60.]); HI = np.array([15, 800, 15, 800, 10, 60, 60.])

def tis(q, F, Ti, L, Tj, C, n, iters=25):
    """n CSTRs in series. n large -> plug flow."""
    lk1, E1, lk2, E2, lh, q1, q2 = q; h = np.exp(lh)
    theta = (L / F) / n
    a = np.ones(len(F)); b = np.zeros(len(F)); T = Ti.copy()
    for _ in range(n):
        a_in, b_in, T_in = a, b, T
        Tk = T_in.copy()
        for _ in range(iters):                       # close the implicit tank
            Tk = np.clip(Tk, 150, 1500)
            inv = 1 / Tk - 1 / Tref
            k1 = np.exp(lk1 - E1 * 1e3 / R * inv)
            k2 = np.exp(lk2 - E2 * 1e3 / R * inv)
            an = a_in / (1 + theta * k1)
            bn = (b_in + theta * k1 * an) / (1 + theta * k2)
            Tn = (T_in + theta * h * Tj + theta * C * (q1 * k1 * an + q2 * k2 * bn)) / (1 + theta * h)
            Tn = np.clip(Tn, 150, 1500)
            if np.max(np.abs(Tn - Tk)) < 1e-9: Tk = Tn; break
            Tk = 0.5 * Tk + 0.5 * Tn                 # damped, keeps stiff rows stable
        inv = 1 / Tk - 1 / Tref
        k1 = np.exp(lk1 - E1 * 1e3 / R * inv); k2 = np.exp(lk2 - E2 * 1e3 / R * inv)
        a = np.clip(a_in / (1 + theta * k1), 0, None)
        b = np.clip((b_in + theta * k1 * a) / (1 + theta * k2), 0, None)
        T = Tk
    return 100 * b

def fit(idx, n, nstart, seed):
    f, t, l, j, c, yy = F[idx], Ti[idx], L[idx], Tj[idx], C[idx], y[idx]
    rng = np.random.default_rng(seed); best = None
    starts = [P_SEED] + [np.clip(P_SEED * rng.uniform(.85, 1.15, 7), LO, HI) for _ in range(nstart)]
    for p0 in starts:
        try:
            r = least_squares(lambda q: tis(q, f, t, l, j, c, n) - yy, p0,
                              bounds=(LO, HI), max_nfev=300)
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    return best.x if best is not None else P_SEED

GRID = [3, 5, 8, 12, 20, 35, 60, 120, 300]
print("STAGE 1 - in-sample fit vs number of tanks   (PFR reference: 3.656)", flush=True)
print("%6s %8s %12s %10s" % ("n", "Pe~2(n-1)", "in-sample", "vs PFR"), flush=True)
res = {}
for n in GRID:
    p = fit(np.arange(len(y)), n, 4, 5)
    e = y - tis(p, F, Ti, L, Tj, C, n)
    rm = float(np.sqrt(np.mean(e ** 2))); res[n] = (rm, p)
    print("%6d %8d %12.4f %10.4f" % (n, 2 * (n - 1), rm, rm - 3.656), flush=True)
    np.save("disp_p_%d.npy" % n, p)

best_n = min(res, key=lambda k: res[k][0])
print("\nbest in-sample n = %d  (RMSE %.4f)" % (best_n, res[best_n][0]), flush=True)
if res[best_n][0] > 3.60:
    print("-> no meaningful in-sample gain over ideal PFR; dispersion does NOT explain the residual", flush=True)

cands = sorted(res, key=lambda k: res[k][0])[:3]
print("\nSTAGE 2 - cross-validating n in %s   (PFR reference CV: 7.36)" % cands, flush=True)
for n in cands:
    oof = np.zeros(len(y))
    for k, (A, B) in enumerate(KF.split(F)):
        p = fit(A, n, 3, k)
        oof[B] = tis(p, F[B], Ti[B], L[B], Tj[B], C[B], n)
    cv = float(np.sqrt(np.mean((oof - y) ** 2)))
    r2 = 1 - np.sum((oof - y) ** 2) / np.sum((y - y.mean()) ** 2)
    print("  n=%3d  Pe~%3d   CV RMSE %7.4f   CV R2 %.4f" % (n, 2 * (n - 1), cv, r2), flush=True)
    np.save("disp_oof_%d.npy" % n, oof)
