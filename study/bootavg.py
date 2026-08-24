"""HEDGING THE BURNOUT CLIFF — bootstrap-averaged predictions.

Problem: ~20 of 50 test rows sit near the burnout cliff, where a small parameter shift
flips the prediction between ~0 and ~65. A single point fit commits to one side.

Solution: RMSE is minimised by the CONDITIONAL MEAN. Resample the 150 training rows,
refit the dispersion model on each resample, and average the test predictions. Where the
model is confident all draws agree (average changes nothing); where it sits on the cliff the
draws disagree and the average lands between them, capping the squared-error loss.

Validated by nested CV: bootstrap inside each training fold, so the comparison against the
single-fit CV of 6.84 is honest.
"""
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

R = 8.314; Tref = 430.0; N_TANKS = 35
tr = pd.read_csv("train_dataset.csv"); te = pd.read_csv("test_dataset.csv")
y = tr.overall_yield.values
def cols(d): return (d.flow_rate_L_min.values, d.inlet_temperature_K.values,
                     d.length_m.values, d.jacket_temperature_K.values, d.concentration_mol_L.values)
F, Ti, L, Tj, C = cols(tr); tF, tTi, tL, tTj, tC = cols(te)
KF = KFold(5, shuffle=True, random_state=0)
INC = np.load("p_final_disp.npy")
LO = np.array([-15, 0, -15, 0, -10, -60, -60.]); HI = np.array([15, 800, 15, 800, 10, 60, 60.])

def tis(q, F, Ti, L, Tj, C, n=N_TANKS, iters=12):
    lk1, E1, lk2, E2, lh, q1, q2 = q; h = np.exp(lh)
    theta = (L / F) / n
    a = np.ones(len(F)); b = np.zeros(len(F)); T = Ti.copy()
    for _ in range(n):
        a_in, b_in, T_in = a, b, T; Tk = T_in.copy()
        for _ in range(iters):
            Tk = np.clip(Tk, 150, 1500); inv = 1/Tk - 1/Tref
            k1 = np.exp(np.clip(lk1 - E1*1e3/R*inv, -50, 50))
            k2 = np.exp(np.clip(lk2 - E2*1e3/R*inv, -50, 50))
            an = a_in / (1 + theta*k1); bn = (b_in + theta*k1*an) / (1 + theta*k2)
            Tn = np.clip((T_in + theta*h*Tj + theta*C*(q1*k1*an + q2*k2*bn)) / (1 + theta*h), 150, 1500)
            if np.max(np.abs(Tn - Tk)) < 1e-8: Tk = Tn; break
            Tk = 0.5*Tk + 0.5*Tn
        inv = 1/Tk - 1/Tref
        k1 = np.exp(np.clip(lk1 - E1*1e3/R*inv, -50, 50)); k2 = np.exp(np.clip(lk2 - E2*1e3/R*inv, -50, 50))
        a = np.clip(a_in / (1 + theta*k1), 0, None)
        b = np.clip((b_in + theta*k1*a) / (1 + theta*k2), 0, None)
        T = Tk
    return 100 * b

def fit(idx, seed):
    f, t, l, j, c, yy = F[idx], Ti[idx], L[idx], Tj[idx], C[idx], y[idx]
    rng = np.random.default_rng(seed); best = None
    for p0 in [INC, np.clip(INC * rng.uniform(.9, 1.1, 7), LO, HI)]:
        try:
            r = least_squares(lambda q: tis(q, f, t, l, j, c) - yy, p0, bounds=(LO, HI), max_nfev=250)
            if best is None or r.cost < best.cost: best = r
        except Exception: pass
    return best.x if best is not None else INC

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
n = len(y); B = 12
t0 = time.time()

# ---- honest nested CV: bootstrap INSIDE each training fold ----
oof_single = np.zeros(n); oof_boot = np.zeros(n)
for k, (A, Bd) in enumerate(KF.split(F)):
    oof_single[Bd] = tis(fit(A, 100 + k), F[Bd], Ti[Bd], L[Bd], Tj[Bd], C[Bd])
    preds = []
    for i in range(B):
        rs = A[np.random.default_rng(2000 + 50*k + i).integers(0, len(A), len(A))]
        preds.append(tis(fit(rs, 300 + i), F[Bd], Ti[Bd], L[Bd], Tj[Bd], C[Bd]))
    oof_boot[Bd] = np.mean(preds, axis=0)
    print("  fold %d done (%.0fs)" % (k, time.time() - t0), flush=True)

cv_s = rmse(np.clip(oof_single, 0, 100), y); cv_b = rmse(np.clip(oof_boot, 0, 100), y)
print("\nsingle fit        CV RMSE %.4f" % cv_s)
print("bootstrap-avg     CV RMSE %.4f   (%+.4f)" % (cv_b, cv_b - cv_s))

# ---- final test predictions ----
boot_p = [fit(np.random.default_rng(9000 + i).integers(0, n, n), 700 + i) for i in range(24)]
P = np.array([np.clip(tis(p, tF, tTi, tL, tTj, tC), 0, 100) for p in boot_p])
pred_boot = P.mean(0); pred_single = np.clip(tis(INC, tF, tTi, tL, tTj, tC), 0, 100)
spread = P.std(0)

print("\nper-row disagreement across bootstrap draws (test set):")
print("  rows with std > 20 : %d   > 10 : %d   > 5 : %d   (of 50)"
      % ((spread > 20).sum(), (spread > 10).sum(), (spread > 5).sum()))
print("  mean |bootstrap-avg - single| = %.2f   max %.2f" %
      (np.abs(pred_boot - pred_single).mean(), np.abs(pred_boot - pred_single).max()))
o = np.argsort(-spread)[:6]
print("\n  most uncertain rows:  single -> bootavg   (std)")
for i in o:
    print("    row %2d:  %6.2f -> %6.2f   (%.1f)" % (i, pred_single[i], pred_boot[i], spread[i]))

np.save("pred_bootavg.npy", pred_boot); np.save("boot_spread.npy", spread)
if cv_b < cv_s:
    pd.DataFrame({"overall_yield": np.round(pred_boot, 3)}).to_csv("submission.csv",
                                                                   index=False, float_format="%.3f")
    print("\n-> bootstrap averaging WINS on CV; submission.csv updated")
else:
    print("\n-> bootstrap averaging does not improve CV; submission left unchanged")
print("total %.0fs" % (time.time() - t0))
