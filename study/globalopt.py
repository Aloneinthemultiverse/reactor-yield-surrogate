"""Is in-sample 3.3881 the GLOBAL optimum, or is the optimiser stuck?

Two independent global searches on the dispersion model (n=35 tanks, Pe~68):
  A) wide random multistart  - 80 starts drawn across the full physical ranges,
                               not perturbed around the incumbent
  B) differential_evolution  - population-based global search, polished with least_squares

If neither beats 3.3881 meaningfully, the incumbent is the global optimum and the
remaining error is model-form, not optimiser failure.
"""
import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
from scipy.optimize import least_squares, differential_evolution

R = 8.314; Tref = 430.0; N_TANKS = 35
tr = pd.read_csv("train_dataset.csv"); y = tr.overall_yield.values
F = tr.flow_rate_L_min.values; Ti = tr.inlet_temperature_K.values
L = tr.length_m.values; Tj = tr.jacket_temperature_K.values; C = tr.concentration_mol_L.values

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
            an = a_in / (1 + theta*k1)
            bn = (b_in + theta*k1*an) / (1 + theta*k2)
            Tn = np.clip((T_in + theta*h*Tj + theta*C*(q1*k1*an + q2*k2*bn)) / (1 + theta*h), 150, 1500)
            if np.max(np.abs(Tn - Tk)) < 1e-8: Tk = Tn; break
            Tk = 0.5*Tk + 0.5*Tn
        inv = 1/Tk - 1/Tref
        k1 = np.exp(np.clip(lk1 - E1*1e3/R*inv, -50, 50))
        k2 = np.exp(np.clip(lk2 - E2*1e3/R*inv, -50, 50))
        a = np.clip(a_in / (1 + theta*k1), 0, None)
        b = np.clip((b_in + theta*k1*a) / (1 + theta*k2), 0, None)
        T = Tk
    return 100 * b

def resid(q): return tis(q, F, Ti, L, Tj, C) - y
def cost(q):
    try:
        r = resid(q); return float(np.sqrt(np.mean(r**2)))
    except Exception: return 1e6

INC = np.load("p_final_disp.npy")
print("incumbent in-sample RMSE %.4f" % cost(INC), flush=True)

# physically sensible global ranges (not perturbations of the incumbent)
LO = np.array([-4.0,   5.0, -6.0,   50.0, -2.0, -40.0, -40.0])
HI = np.array([ 8.0, 150.0,  6.0,  500.0,  4.0,  40.0,  40.0])

# ---------- A) wide random multistart ----------
t0 = time.time(); rng = np.random.default_rng(4242)
best_a, best_ax = cost(INC), INC.copy()
for i in range(80):
    p0 = rng.uniform(LO, HI)
    try:
        r = least_squares(resid, p0, bounds=(LO, HI), max_nfev=200)
        c = float(np.sqrt(np.mean(r.fun**2)))
        if c < best_a - 1e-9: best_a, best_ax = c, r.x; print("   start %2d -> %.4f" % (i, c), flush=True)
    except Exception: pass
print("A) wide multistart best %.4f   (%.0fs)" % (best_a, time.time()-t0), flush=True)

# ---------- B) differential evolution ----------
t0 = time.time()
de = differential_evolution(cost, list(zip(LO, HI)), seed=7, maxiter=40, popsize=10,
                            tol=1e-8, mutation=(0.4, 1.0), recombination=0.8, polish=False)
pol = least_squares(resid, np.clip(de.x, LO, HI), bounds=(LO, HI), max_nfev=400)
best_b = float(np.sqrt(np.mean(pol.fun**2)))
print("B) differential_evolution %.4f -> polished %.4f   (%.0fs)" % (de.fun, best_b, time.time()-t0), flush=True)

winner, wx = min([(best_a, best_ax), (best_b, pol.x), (cost(INC), INC)], key=lambda t: t[0])
print("\nBEST OVERALL in-sample %.4f   (incumbent was %.4f)" % (winner, cost(INC)), flush=True)
if winner < cost(INC) - 0.05:
    print("-> optimiser WAS stuck; better optimum found"); np.save("p_globalbest.npy", wx)
    print("   params", np.round(wx, 4))
else:
    print("-> incumbent confirmed as global optimum; residual is MODEL-FORM error, not optimiser failure")
