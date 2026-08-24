# How this model works — taught from zero

A complete walkthrough for someone who knows a little machine learning and no chemistry.
Every term is defined before it is used. Take it slowly; each part builds on the last.

---

## Part 0 — What you need to know already

Only three things:

1. **Supervised learning** — you have examples with known answers, and you want to predict the
   answer for new examples.
2. **Training** — adjusting a model so its predictions match the known answers.
3. **Overfitting** — when a model memorises the training examples instead of learning the
   pattern, so it fails on new ones.

That's it. Everything else is explained below.

---

# Part 1 — The problem

## 1.1 A reactor is a heated pipe

Imagine a long metal pipe. Liquid chemical flows in one end and out the other. Wrapped around
the pipe is a heating jacket — like a blanket with hot water in it — that controls the
temperature inside.

```
          ┌─────── heating jacket (T_jacket) ───────┐
   A ───→ │═════════════════════════════════════════│ ───→ mixture out
          └─────────── length L ────────────────────┘
```

While the liquid travels down the pipe, chemistry happens.

## 1.2 The chemistry: A becomes B becomes C

```
   A   ──────→   B   ──────→   C
 (feed)      (want this!)   (worthless)
```

Chemical **A** goes in. Inside the pipe, A slowly turns into **B**. But B doesn't stop — B
keeps reacting and turns into **C**, which is useless.

**This is the entire difficulty.** You want to stop at B.

### The cookie analogy — remember this one

```
   dough  ──────→  cookie  ──────→  burnt
    (A)             (B)              (C)
```

Take them out too early → still dough. Too late → burnt. There is a perfect moment.

## 1.3 The five things you can control

| Input | Symbol | What it does |
|---|---|---|
| Flow rate | `F` | How fast you pump. Faster = less time inside. |
| Concentration | `C₀` | How much A is in the feed. |
| Inlet temperature | `T_in` | How hot the liquid is when it enters. |
| Length | `L` | How long the pipe is. |
| Jacket temperature | `T_jacket` | How hot the heating blanket is. |

## 1.4 The one thing you want to predict

**Overall yield of B** — what percentage of the A you fed in came out as B.

- 100 % = perfect, everything became B
- 0 % = disaster, everything became C (or nothing reacted)

## 1.5 The competition

They gave us:

- **150 training rows** — five inputs *and* the answer
- **50 test rows** — five inputs only. Predict the answers.

Scored by **RMSE** (explained in Part 6). Lower is better. 374 teams competed.

### Why anyone cares

Engineers *can* compute the exact answer by solving the physics equations on a computer — but
it takes **minutes to hours per calculation**. Far too slow to run a live factory, where you
want to adjust settings and see the consequence immediately.

A fast model that gives the same answer in **milliseconds** is called a **surrogate model**.
That's what we were asked to build.

---

# Part 2 — What normal ML would do, and why it struggles

## 2.1 The standard recipe

Most teams would do this:

```python
model = XGBoost()
model.fit(X_train, y_train)      # 150 rows, 5 columns
predictions = model.predict(X_test)
```

The model looks for patterns: *"when temperature is high, yield tends to be low."* It doesn't
know what a reactor is. It's pattern-matching.

**We tried this.** Results:

| Model | Error (lower = better) |
|---|---|
| HistGradientBoosting | 22.87 |
| ExtraTrees | 17.01 |

## 2.2 Why it struggles — three reasons

### Reason 1: only 150 examples

A tree ensemble makes decisions like *"if temperature > 437, go left."* Each such decision is
a number learned from the data. A forest has **thousands** of them.

> **Estimating thousands of numbers from 150 examples is like drawing a detailed map of a city
> after glancing at 150 street corners.** You'll get the corners right and invent the rest.

### Reason 2: trees draw staircases, the truth is a smooth curve

A decision tree can only output constants inside boxes. Asked to model a smooth curve, it
produces a staircase:

```
  truth (smooth)          what a tree gives you
       ___                      ____
      /                        _|
     /                       _|
    /                      _|
   /                     _|
```

The real relationship here involves `exp()` — genuinely smooth and steep. A staircase
approximates it badly, especially with few steps to spare.

### Reason 3: there is a cliff

Above about 470 K, yield doesn't decline gradually — it **collapses to zero**. In our training
data, 96–100 % of rows above 470 K have *exactly* zero yield.

A tree can represent a cliff, but it needs examples on both sides to find the edge. With 150
rows spread over five dimensions, there aren't many near the edge.

---

# Part 3 — The key idea

## 3.1 The realisation

The problem statement said the data came from **physics simulations**.

That means the answers weren't measured in a lab — someone wrote a program that solves the
reactor equations, ran it 200 times, and gave us the results.

Two consequences:

1. **There is no noise.** Every answer is exactly what the equations produce. No measurement
   error to average out.
2. **A perfect answer exists.** If we could write down the *same equations* with the *same
   constants*, we'd predict every row exactly.

## 3.2 So we changed the question

> **Wrong question:** which ML algorithm best fits these 150 points?
>
> **Right question:** what equations generated these 150 points?

This is **system identification** — figuring out the rules of a system by watching what it
does. Detective work rather than curve-fitting.

## 3.3 What we still don't know

We know the *form* of the physics (it's textbook chemical engineering). We don't know the
**constants** — the specific numbers for this particular reaction.

**So: write the equations with the constants left blank, then use the 150 examples to fill in
the blanks.** That last step is the machine learning.

---

# Part 4 — Building the model, one piece at a time

We add one idea at a time. Watch the error drop as each is added.

## 4.1 Piece one: residence time

Flow rate and length don't matter separately. What matters is **how long the liquid is inside**:

```
τ = L / F           (τ is "tau", the Greek letter — standard notation)
```

Long pipe, slow pump → lots of time. Short pipe, fast pump → very little.

**Example:** row 1 of the test set, L = 19.95 m, F = 47.80 L/min → **τ = 0.417**

> 💡 Five inputs, but the physics only sees four things: τ, C₀, T_in, T_jacket.

## 4.2 Piece two: reaction rates

How fast does A turn into B? That's a **rate constant**, `k₁`. Similarly `k₂` for B→C.

```
speed of A→B = k₁ × (how much A is present)
speed of B→C = k₂ × (how much B is present)
```

Bigger k = faster reaction.

## 4.3 Piece three: rates depend on temperature (Arrhenius)

This is the most important equation in chemistry. Heat makes reactions faster — but not
linearly. **Exponentially.**

```
k = k_ref · exp[ −Ea/R · (1/T − 1/T_ref) ]
```

Don't panic. Reading it piece by piece:

| Symbol | Meaning |
|---|---|
| `T` | Temperature (in Kelvin) |
| `Ea` | **Activation energy** — how temperature-sensitive this reaction is |
| `R` | 8.314, a universal constant of nature |
| `k_ref`, `T_ref` | A reference point: "the rate is `k_ref` when the temperature is `T_ref`" |

**What `Ea` controls:**

- Small `Ea` → temperature barely matters
- Large `Ea` → temperature matters enormously

**Concrete:** heat the reactor by 10 K, from 440 to 450:

| Reaction | Ea | Rate change |
|---|---|---|
| A→B (desired) | 45 kJ/mol | ×1.3 — mild |
| B→C (side) | 252 kJ/mol | **×4.6** — dramatic |

> 🔑 **This single fact is the whole problem.** Heating speeds up the reaction you *want* a
> little, and the one you *don't want* a lot.

## 4.4 Piece four: what happens along the pipe

We now track three things as the liquid travels:

- `a` = fraction still A (starts at 1.0)
- `b` = fraction now B (starts at 0.0)
- `T` = temperature (starts at T_in)

Three rules:

```
① da/dt = −k₁·a
```
*A disappears. The more A you have, the faster it goes.*

```
② db/dt = k₁·a − k₂·b
           ↑gain  ↑loss
```
*B is created from A **and** destroyed into C, at the same time.*

> 🔑 **Equation ② is the competition in one line.** When the loss term overtakes the gain term,
> yield starts falling.

```
③ dT/dt = h·(T_jacket − T) + C₀·(q₁·k₁·a + q₂·k₂·b)
           └─ jacket ─┘      └── heat from reactions ──┘
```
*Temperature changes for two reasons: the jacket heats or cools the liquid, and the reactions
themselves release or absorb heat.*

New symbols: `h` = how well the jacket transfers heat; `q₁`, `q₂` = heat released by each
reaction (negative = absorbs heat).

## 4.5 Piece five: solving it in 35 slices

Those three rules describe *change*. To get an actual answer, we walk down the pipe in small
steps.

Chop the reactor into **35 slices**. Each gets `θ = τ/35` of reaction time. For each slice:

1. Look at the current temperature `T`
2. Compute `k₁` and `k₂` from Arrhenius
3. React a little A into B, a little B into C
4. Update `T` from the jacket and reaction heat
5. Hand `(a, b, T)` to the next slice

After 35 slices, whatever `b` remains **is the answer**.

### A subtlety: each slice needs iteration

Temperature and reaction depend on *each other*. The rate depends on T, but T changes because
of the reaction. Which do you compute first?

Neither. You **iterate until they agree**:

```
guess T → compute rates → compute the T that implies → did it change?
   ↑                                                          │
   └──────────────── yes, guess again ────────────────────────┘
```

Converges in 5–10 passes. This is called an **implicit** solve, and it's why the model is
stable where a naive approach blows up.

---

# Part 5 — Fitting: where the machine learning happens

We now have equations with **7 blanks**: `k₁, Ea₁, k₂, Ea₂, h, q₁, q₂`.

## 5.1 Linear regression — and why it doesn't apply

You may have seen:

```
y = β₀ + β₁x
```

Fitting this is easy — there's a **formula** that gives the exact best answer in one step.

Here's the thing most people get wrong: **"linear regression" means linear in the *parameters*,
not the inputs.** All of these are still linear regression:

```
y = β₀ + β₁x + β₂x²
y = β₀ + β₁·log(x)
```

They're curves, but each parameter is just a multiplier. Double `β₂`, and that term's
contribution doubles. That property is what makes the one-step formula work.

## 5.2 Why ours is *non-linear* regression

Look at Arrhenius again:

```
k = k_ref · exp[ −Ea/R · (1/T − 1/T_ref) ]
```

**`Ea` is inside the exponential.** Double `Ea` and `k` does not double — it changes by an
exponential factor that depends on T. You cannot factor `Ea` out.

**No formula exists.** You have to *search* for the answer.

It's worse than that: our model isn't even an equation. It's *35 iterations of a coupled
solve*. There's no algebraic expression relating `yield` to `Ea₂`. The only way to find out
what `Ea₂ = 250` predicts is to run the whole simulation.

## 5.3 How the search works

```
1. Guess 7 numbers
2. Run the simulation on all 150 training rows
3. Compare predictions to the known answers
4. Add up the squared differences  →  "loss"
5. Work out which direction reduces the loss
6. Take a step that way
7. Repeat until it stops improving
```

Step 5 is done by **Levenberg–Marquardt**, which blends two strategies:

| Strategy | Behaviour |
|---|---|
| Gradient descent | Small careful step downhill. Slow but safe. |
| Gauss–Newton | Pretend it's locally simple, jump to the answer. Fast but reckless. |

LM automatically uses gradient descent when far away and Gauss–Newton when close. In scipy:

```python
result = least_squares(residual_function, initial_guess, bounds=(lo, hi))
```

## 5.4 The catch: local minima

Linear regression has exactly one answer. Non-linear can have many — you might get stuck in a
small dip instead of the true bottom.

```
loss
  │  ╲        ╱╲
  │   ╲      ╱  ╲      ← stuck here (local minimum)
  │    ╲╱╲  ╱    ╲
  │        ╲╱     ╲___  ← want this (global minimum)
  └────────────────────→ parameter value
```

**How we handled it:** started the search from many random guesses. Then verified with
`differential_evolution` (a global search that explores broadly before refining). Both landed
on the same answer — in-sample error **3.3911** — so we know it's the true bottom, not a dip.

## 5.5 The answer

```
Ea₁ =  45.1 kJ/mol      A → B   (desired reaction)
Ea₂ = 251.9 kJ/mol      B → C   (side reaction)
h   =  3.41             heat transfer from the jacket
q₁  = −13.33            A → B absorbs heat (endothermic)
q₂  = +11.76            B → C releases heat (exothermic)
```

**Seven numbers. That's the whole model.**

---

# Part 6 — Checking we didn't fool ourselves

## 6.1 The trap

If you judge a model by how well it fits the data you trained on, you will pick a model that
**memorised** rather than learned.

**We proved this on our own data.** One variant we tested had the *best training score in the
entire study* — 3.33. Its cross-validated score was **41.2**, with a **negative R²**, meaning
worse than ignoring the inputs and always guessing the average.

> ⚠️ Had we selected on training error, we would have submitted the worst model we built.

## 6.2 Cross-validation

The fix: hide some data from yourself.

```
Split 150 rows into 5 groups of 30.

Round 1: train on groups 2,3,4,5 → predict group 1
Round 2: train on groups 1,3,4,5 → predict group 2
... 5 rounds ...

Every row gets predicted by a model that never saw it.
```

That score is an honest estimate of performance on new data.

## 6.3 RMSE and R²

**RMSE** (Root Mean Squared Error):

```
RMSE = √( average of (true − predicted)² )
```

Same units as yield, so RMSE = 5 means "typically off by about 5 percentage points". Squaring
means **one huge mistake hurts far more than several small ones** — that matters in Part 8.

**R²** — fraction of the variation explained. 1.0 = perfect, 0 = no better than guessing the
average, **negative = worse than guessing the average**.

## 6.4 Bagging

Our final trick. Instead of fitting once, we fit **32 times**, each on a random resample of
the 150 rows, and average the predictions.

This is **bagging** (bootstrap aggregating) — the same idea underneath Random Forest.

**Why it helps:** where the reactor is well-behaved, all 32 fits agree and averaging changes
nothing. Near the cliff they disagree — one says 60 %, another says 5 % — and the average
lands between. Since RMSE punishes large errors quadratically, hedging beats committing.

---

# Part 7 — Results

## 7.1 The ladder

Each row adds one physical idea:

| Model | CV error |
|---|---|
| HistGradientBoosting | 22.87 |
| ExtraTrees | 17.01 |
| Physics: A→B→C, basic | 9.75 |
| **+ reaction heat** | 7.36 |
| **+ axial dispersion** | 6.84 |
| **+ bagging & thermal correction** | **≈ 4.8** |

About **3.5× better** than the best ML baseline.

## 7.2 What we rejected, and why

| Idea tested | Training error | CV error | Verdict |
|---|---|---|---|
| Parallel path A→C | 8.28 | 10.16 | ❌ optimiser set its rate to zero |
| Free reaction orders | **3.33** | **41.2** | ❌ overfitting |
| Flow-dependent heat transfer | 3.64 | 7.78 | ❌ made things worse |

## 7.3 The result we're proudest of

Our fitted constants say `k₂ = k₁` at **449.1 K**. Below that, B survives. Above it, B is
destroyed faster than it forms.

That number came purely from the model. Now look at the **raw data**, no model involved:

| Temperature band | Average yield |
|---|---|
| 400–430 K | 56.9 % |
| 430–460 K | 23.7 % |
| 460–490 K | **0.2 %** |

The data collapses across exactly the band our equation predicted.

> 🔑 **Two completely independent routes — a fitted equation and a raw correlation — agree on
> the same threshold.** That's the strongest evidence that the constants are physically real
> and not just numbers that happened to fit.

---

# Part 8 — Where it fails

A model presented as flawless is a model nobody should trust.

## 8.1 Our leaderboard result

```
RMSE = 10.20        MAE = 2.65
```

**MAE** (Mean Absolute Error) is the average error *without* squaring.

Our MAE of 2.65 was better than the team that placed **3rd**. So on a typical row we were more
accurate than a team two places above us. Why was our RMSE worse?

Because RMSE squares. Solving for the error pattern that produces both numbers:

> **≈ 2 rows wrong by ~50 points, and ≈ 48 rows wrong by ~0.6 points.**

Not a mediocre model. A near-exact model that fell off a cliff twice.

## 8.2 Where those two rows sit

Both are within **2 K of the 449.1 K crossover** — the exact temperature where k₂ overtakes
k₁.

And our uncertainty estimate had flagged one of them as the **least reliable prediction in the
entire test set — before we submitted.**

## 8.3 Why that region is genuinely brutal

The optimal residence time collapses as you approach the crossover:

| Temperature | τ_opt |
|---|---|
| 380 K | 3.16 min |
| 420 K | 0.32 min |
| 450 K | **0.03 min** |

**A 100× change.** Sitting on the crossover means the perfect operating point moves faster
than any measurement can track. That's not a modelling defect — it's a physically
ill-conditioned regime, and it would defeat a real operator too.

## 8.4 If this were deployed in a real plant

**Chemical causes:** fouling (deposits on the pipe wall reduce `h`), catalyst degradation,
feed composition drift, loss of steady state.

**Data/ML causes:** operating outside the 350–500 K training range, sensor drift on the jacket
thermocouple, extrapolation.

**How would you tell which?** This is where physical parameters pay off. Refit the model on
recent plant data and look at *which constant moved*:

| What drifted | What it means |
|---|---|
| `h` | Fouling or scaling — a heat-transfer problem |
| `Ea` | Catalyst chemistry has changed |
| Nothing, but residuals are structured | The model itself is wrong |

**A black box cannot do this.** If XGBoost starts failing, all you know is that it's failing.

---

# Part 9 — Glossary

| Term | Meaning |
|---|---|
| **Surrogate model** | A fast approximation of a slow simulation |
| **Yield** | Percentage of feed that became the desired product |
| **Residence time (τ)** | How long liquid stays in the reactor = L/F |
| **Rate constant (k)** | How fast a reaction goes |
| **Arrhenius equation** | How rate constants depend on temperature (exponentially) |
| **Activation energy (Ea)** | How temperature-sensitive a reaction is |
| **Endothermic / Exothermic** | Absorbs heat / releases heat |
| **Non-isothermal** | Temperature varies along the reactor |
| **ODE** | An equation describing how something changes |
| **Linear regression** | Model linear in its *parameters* — has a one-step formula |
| **Non-linear regression** | Parameters inside exponentials etc. — must be searched for |
| **Levenberg–Marquardt** | The search algorithm we used |
| **Local minimum** | A dip that isn't the true bottom |
| **Cross-validation** | Testing on data hidden from training |
| **RMSE / MAE** | Error with / without squaring |
| **R²** | Fraction of variation explained; negative = worse than guessing |
| **Bagging** | Fitting many times on resamples and averaging |
| **Overfitting** | Memorising the training data instead of the pattern |

---

# Part 10 — The whole thing in ten sentences

1. A reactor turns A into B, but B keeps turning into C, so yield rises then falls.
2. We had 150 examples and had to predict 50 more.
3. Standard ML got an error around 17 because 150 rows is too few to learn a smooth
   exponential surface with a cliff in it.
4. The data came from a physics simulator, so instead of learning patterns we tried to rebuild
   the simulator.
5. We wrote the textbook reactor equations with 7 constants left blank.
6. We searched for the 7 constants that best explain the 150 examples — non-linear regression,
   because activation energy sits inside an exponential.
7. We chose which physical effects to include by cross-validation, and rejected three of them,
   including one that had the best training score and a negative CV R².
8. The fitted constants say the reactor stops working above 449.1 K — and the raw data
   independently collapses at exactly that temperature.
9. Our final error was about 4.8 in cross-validation, roughly 3.5× better than the best ML
   baseline, and 6th of 374 on the leaderboard.
10. Our two big failures both sit within 2 K of that 449 K cliff, and our uncertainty estimate
    had flagged one of them before we submitted.
