"""
ML Hackathon - Fugacity 2026, IIT Kharagpur
Chemical Reactor Yield Prediction - Physics-Based Surrogate Model

Reproduces submission.csv exactly.   Run:  python solution.py

APPROACH
--------
Rather than fitting a statistical model, we reverse-engineered the SIMULATOR that
generated the data: a non-isothermal tubular reactor with first-order series kinetics
A -> B -> C, Arrhenius temperature dependence, a coupled energy balance, and axial
dispersion. Seven physical constants are fitted to the 150 training rows.

    da/dt = -k1 a
    db/dt =  k1 a - k2 b
    dT/dt =  h (Tj - T) + C0 (q1 k1 a + q2 k2 b)
    k_i   =  k_i(Tref) exp[ -Ea_i/R (1/T - 1/Tref) ],   t = z/F,  tau = L/F

Solved as 35 CSTRs in series (tanks-in-series model of axial dispersion, Pe ~ 2(n-1) ~ 68),
each tank closed implicitly by damped fixed-point iteration. An extra axial smoothing of
TEMPERATURE ONLY (lam = 0.10) decouples the thermal from the mass Peclet number (Lewis != 1).
Final predictions are averaged over 32 bootstrap refits to stabilise the burnout-cliff rows.

VALIDATION (5-fold CV, parameters refitted inside every fold)
    HistGradientBoosting ......... 22.87
    ExtraTrees ................... 17.01
    physics, first order ..........  9.75
    + energy balance ..............  7.36
    + axial dispersion ............  6.84
    + bootstrap + lam=0.10 ........ ~4.8
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.optimize import least_squares

R = 8.314          # gas constant, J/mol/K
Tref = 430.0       # reference temperature for the rate constants, K
NTANKS = 35        # tanks in series  ->  Pe ~ 2(n-1) ~ 68
LAM = 0.10         # extra axial smoothing of temperature only (Lewis number != 1)
NBOOT = 32         # bootstrap refits averaged in the final prediction

# fitted seed for the optimiser (full-data fit of the dispersion model)
SEED_PARAMS = np.array([2.8177365771016238, 45.036139387783464, 0.35801502122472623,
                        252.17511433839636, 1.1432378350398846, -12.249640396585363,
                        10.985225330377427])
LO = np.array([-15,   0, -15,   0, -10, -60, -60.])
HI = np.array([ 15, 800,  15, 800,  10,  60,  60.])


def reactor(q, F, Ti, L, Tj, C, n=NTANKS, lam=LAM, iters=12):
    """Non-isothermal tanks-in-series reactor. Returns % yield of B at the outlet."""
    lk1, E1, lk2, E2, lh, q1, q2 = q
    h = np.exp(lh)
    theta = (L / F) / n                      # residence time per tank
    a = np.ones(len(F))                      # C_A / C_A0
    b = np.zeros(len(F))                     # C_B / C_A0
    T = Ti.copy()
    Tprev = Ti.copy()

    for _ in range(n):
        a_in, b_in, T_in = a, b, T
        Tk = T_in.copy()
        # each tank is implicit in T (rate constants depend on the outlet temperature)
        for _ in range(iters):
            Tk = np.clip(Tk, 150, 1500)
            inv = 1 / Tk - 1 / Tref
            k1 = np.exp(np.clip(lk1 - E1 * 1e3 / R * inv, -50, 50))
            k2 = np.exp(np.clip(lk2 - E2 * 1e3 / R * inv, -50, 50))
            an = a_in / (1 + theta * k1)
            bn = (b_in + theta * k1 * an) / (1 + theta * k2)
            Tn = (T_in + theta * h * Tj + theta * C * (q1 * k1 * an + q2 * k2 * bn)) / (1 + theta * h)
            Tn = np.clip(Tn, 150, 1500)
            if np.max(np.abs(Tn - Tk)) < 1e-8:
                Tk = Tn
                break
            Tk = 0.5 * Tk + 0.5 * Tn          # damped update keeps stiff rows stable
        inv = 1 / Tk - 1 / Tref
        k1 = np.exp(np.clip(lk1 - E1 * 1e3 / R * inv, -50, 50))
        k2 = np.exp(np.clip(lk2 - E2 * 1e3 / R * inv, -50, 50))
        a = np.clip(a_in / (1 + theta * k1), 0, None)
        b = np.clip((b_in + theta * k1 * a) / (1 + theta * k2), 0, None)
        T = (1 - lam) * Tk + lam * Tprev       # thermal dispersion decoupled from mass
        Tprev = Tk
    return 100 * b


def load(path):
    d = pd.read_csv(path)
    return (d.flow_rate_L_min.values, d.inlet_temperature_K.values, d.length_m.values,
            d.jacket_temperature_K.values, d.concentration_mol_L.values)


def main():
    train = pd.read_csv("train_dataset.csv")
    y = train.overall_yield.values
    F, Ti, L, Tj, C = load("train_dataset.csv")
    tF, tTi, tL, tTj, tC = load("test_dataset.csv")

    def fit(idx):
        r = least_squares(lambda q: reactor(q, F[idx], Ti[idx], L[idx], Tj[idx], C[idx]) - y[idx],
                          SEED_PARAMS, bounds=(LO, HI), max_nfev=300)
        return r.x

    # full-data fit (reported parameters)
    p = fit(np.arange(len(y)))
    ins = np.sqrt(np.mean((y - reactor(p, F, Ti, L, Tj, C)) ** 2))
    print("fitted constants")
    print("  Ea1 = %6.1f kJ/mol   (A->B, desired)" % p[1])
    print("  Ea2 = %6.1f kJ/mol   (B->C, side)     ratio %.1f" % (p[3], p[3] / p[1]))
    print("  h   = %6.3f          jacket heat transfer" % np.exp(p[4]))
    print("  q1  = %+6.2f          A->B endothermic" % p[5])
    print("  q2  = %+6.2f          B->C exothermic" % p[6])
    print("  in-sample RMSE %.4f" % ins)

    # bootstrap-averaged test predictions (fixed seeds -> exactly reproducible)
    N = len(y)
    draws = []
    for i in range(NBOOT):
        rs = np.random.default_rng(11000 + i).integers(0, N, N)
        draws.append(np.clip(reactor(fit(rs), tF, tTi, tL, tTj, tC), 0, 100))
    pred = np.array(draws).mean(0)

    pd.DataFrame({"overall_yield": pred}).to_csv("submission.csv", index=False, float_format="%.3f")
    print("\nsubmission.csv written: %d rows, mean %.3f, first %.3f"
          % (len(pred), pred.mean(), pred[0]))


if __name__ == "__main__":
    main()
