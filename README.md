# EEG Structured Noise Lab

### Statistical inverse modelling for EEG source localization

This project asks a simple question:

> **If measurement uncertainty in EEG has spatial and temporal structure, can explicitly modelling that structure improve inverse source localization?**

The project starts from a deliberately simple first-principles simulator, develops progressively more realistic inverse problems, analyses real EEG residuals, and finally transfers an empirical correlation structure observed in real EEG into a controlled simulator.

The goal is **not** to assume that EEG noise is definitely structured.

Instead, the project asks whether a structured-noise model is worth considering when the unexplained variability in measurements contains reproducible correlations.

---

## 1. The intuition

Imagine trying to locate a hidden electrical source inside the brain using measurements from electrodes on the scalp.

You cannot see the source directly.

You only see the measurements it produces.

So the problem looks roughly like this:

```text
Hidden source
     ↓
Electrical field
     ↓
EEG electrodes
     ↓
Measured signals
     ↓
Inverse problem
     ↓
Estimated source location
```

The difficult part is that the measurements are never perfectly clean.

There is always unexplained variation.

A simple model might treat that variation as independent noise:

**y = Gx + ε**

where:

- **y** is the measured EEG
- **x** is the unknown source
- **G** is the forward model connecting source to electrodes
- **ε** represents measurement uncertainty

The usual simplified assumption is:

**ε ~ N(0, σ²I)**

which essentially says:

> each electrode gets its own random error, with the same variance, and those errors are independent.

But what if that assumption is too simple?

If two nearby electrodes tend to fluctuate together, or if some measurement deviations repeatedly appear together across trials, then the uncertainty is no longer completely independent.

A more general model is:

**ε ~ N(0, Σ)**

where **Σ** describes how the uncertainty varies and how different measurements are related.

That leads to the central idea of this project:

> **Instead of treating all unexplained measurement variation as one undifferentiated noise process, can a structured covariance model make the inverse problem more informed?**

---

# 2. What this project is actually testing

The research question is:

> **Can probabilistic spatiotemporal modelling of a structured noise component improve inverse source localization compared with treating measurement uncertainty as a single undifferentiated noise process?**

There are two important distinctions here.

### This project does not claim:

- that all EEG noise is structured
- that EEG channel correlation automatically represents noise
- that real EEG experiments prove improved source localization
- that covariance-aware localization is a new idea
- that the real EEG dataset provides a known ground-truth source

Instead, the project separates the problem into controlled and observational stages.

### Controlled simulation

The simulator provides a known source location.

That means localization accuracy can actually be measured.

### Real EEG

Real EEG is used to ask a different question:

> **Do reproducible spatial relationships exist in trial-to-trial residual variability?**

The two pieces are then connected by transferring the empirically observed correlation structure into the controlled simulator.

---

# 3. The modelling idea

For a candidate source **x**, the forward model predicts what the electrodes should measure:

**ŷ = Gx**

The inverse problem compares the actual measurement **y** against predictions from many possible source locations.

With independent-noise assumptions, the project uses an ordinary error measure:

**RMSE(y, ŷ)**

The candidate with the smallest error is selected.

With a covariance model, the residual is evaluated using:

**eᵀΣ⁻¹e**

where:

**e = y − ŷ**

This changes the question from:

> “How large is the error?”

to:

> “How surprising is this error given the expected structure of the uncertainty?”

That distinction is the core mathematical idea behind the project.

---

# 4. Building the problem from first principles

Rather than starting immediately with a large EEG library, the project was developed progressively.

### V0 — 2D toy forward/inverse model

A circular head-like geometry was created with electrodes around its boundary.

A source was placed at a known position.

The simulator generated electrode measurements from the source.

Then every possible candidate location was tested to see whether the inverse procedure could recover the original source.

**Purpose:** understand the inverse problem itself before introducing complicated data.

---

### V1 — Independent Gaussian measurement noise

Random Gaussian measurement noise was added to the measurements.

The experiment was repeated many times.

The important observation was:

> Independent noise changes individual measurements, but under the tested setup it does not create a reason for a covariance-aware method to outperform ordinary RMSE.

This became the control experiment.

---

### V1.5 — Off-grid source

The source was moved away from the discrete candidate grid.

This introduced a more realistic problem:

> the true source does not necessarily have to be one of the locations the inverse algorithm is allowed to choose.

---

### V2 — Dipole orientation

The source was extended from a scalar point-like source to a dipole with orientation.

This introduced direction into the forward model.

---

### V3 — 3D spherical model

The simulator was extended into three dimensions.

It uses:

- a spherical geometry
- multiple electrodes
- a dipole source
- a discrete 3D candidate grid
- a physically motivated dipole forward model

This became the foundation for the controlled statistical experiments.

---

# 5. Independent-noise control

The first major experiment compares two localization methods under independent Gaussian noise.

The experiment uses:

- sphere radius: 3
- 8 electrodes
- true source: **[1, 2, 1]**
- 93 candidate source locations
- noise standard deviation: 0.1
- 10,000 calibration trials
- 10,000 localization trials
- fixed random seed: 42

### Result

| Method | Exact recovery |
|---|---:|
| RMSE localization | 94.12% |
| Covariance-aware localization | 94.12% |

The localization errors were identical across all 10,000 trials.

That is exactly the kind of control the project needed.

If the covariance-aware method had improved performance here, it would have suggested that the comparison was being driven by something other than structured covariance.

Instead:

> **Under independent noise, both methods behave identically in the tested setup.**

---

# 6. Introducing structured covariance

The next experiment deliberately introduces spatially correlated uncertainty.

Instead of:

**Σ = σ²I**

the simulator constructs a covariance matrix whose entries depend on electrode separation:

**Σᵢⱼ = σ² exp(−dᵢⱼ / ℓ)**

where:

- **dᵢⱼ** is the distance between electrodes
- **ℓ** is a correlation-length parameter
- **σ** controls the overall noise scale

Intuitively:

> electrodes that are closer together are allowed to experience more similar measurement fluctuations.

The experiment was repeated across multiple correlation lengths and multiple random seeds.

### Main pattern

As the imposed spatial correlation became stronger, covariance-aware localization increasingly outperformed ordinary RMSE localization.

For the tested configuration:

| Correlation length | Mean accuracy improvement |
|---:|---:|
| 0.5 | ~0 pp |
| 1 | ~0 pp |
| 2 | +0.81 pp |
| 4 | +3.88 pp |
| 8 | +6.89 pp |

At correlation length 8, all 10 tested random seeds produced an improvement.

This does **not** establish that real EEG has the same covariance structure.

It establishes something narrower:

> **When the measurement uncertainty actually contains the type of spatial structure encoded by the covariance model, explicitly modelling that structure can improve localization under the tested simulator.**

---

# 7. Looking at real EEG

The next question was:

> Is there evidence in real EEG that trial-to-trial residual variability contains reproducible spatial structure?

The project uses the MNE sample dataset and focuses on an event-locked auditory condition.

The dataset contains:

- 72 matching trials
- 59 usable EEG channels
- 421 time points per epoch
- epochs from −0.2 s to +0.5 s around the event

For each trial, the across-trial average response is removed:

**rₜᵣᵢₐₗ(t) = yₜᵣᵢₐₗ(t) − ȳ(t)**

These residuals represent the trial-to-trial variability that remains after removing the average event-related response.

An important scientific distinction:

> **These residuals are not assumed to be pure noise.**

They contain whatever variability is not explained by the averaged response, including measurement noise and potentially physiological or other sources of variation.

---

# 8. Is the observed covariance reproducible?

A real covariance matrix by itself is not enough.

A correlation can appear simply because of finite sampling.

So the project performs a reproducibility test.

Real EEG residuals are compared against a shuffled null construction that disrupts the original trial structure.

The difference between the real and null covariance structure is examined at several points in the epoch.

### Selected results

| Time | Real covariance reproducibility | Shuffled null | Difference |
|---:|---:|---:|---:|
| −0.200 s | 0.2579 ± 0.0539 | −0.0112 ± 0.0171 | 0.2691 |
| 0.000 s | 0.5977 ± 0.0297 | 0.0007 ± 0.0145 | 0.5970 |
| +0.133 s | 0.5552 ± 0.0165 | −0.0017 ± 0.0236 | 0.5569 |
| +0.300 s | 0.1536 ± 0.0219 | −0.0090 ± 0.0223 | 0.1627 |
| +0.499 s | 0.8212 ± 0.0294 | 0.0054 ± 0.0232 | 0.8158 |

The real residual covariance structure is substantially more reproducible than the shuffled null in these tested windows.

This supports the narrower observation that:

> **trial-to-trial EEG residual variability contains reproducible spatial covariance structure in this dataset and analysis.**

It does not, by itself, prove that this structure is exclusively measurement noise.

---

# 9. The real EEG → simulator bridge

This is the final controlled experiment.

The real EEG analysis identifies an empirical correlation structure.

That structure is then transferred into the simulator.

The bridge covariance is constructed as:

**Σ_bridge = σ²_sim R_EEG**

where:

- **R_EEG** is the correlation structure estimated from real EEG residuals
- **σ_sim** sets the simulator's noise scale

The simulator still knows the true source.

Therefore, unlike real EEG source localization, the localization error can be measured exactly.

---

# 10. Final bridge result

Using the empirical real-EEG correlation structure in the controlled simulator:

| Method | Exact localization accuracy |
|---|---:|
| RMSE | 91.40% ± 0.28% |
| Covariance-aware | 100.00% ± 0.00% |

### Improvement

**+8.60 percentage points**

Mean localization error:

| Method | Mean error |
|---|---:|
| RMSE | 0.103069 |
| Covariance-aware | 0.000010 |

Mean error reduction:

**0.103059**

The important interpretation is not:

> “Real EEG localization is 100% accurate.”

It is:

> **When the empirical spatial correlation structure observed in real EEG residuals is transferred into a controlled simulator, covariance-aware localization substantially outperforms ordinary RMSE under the tested conditions.**

The simulator provides the ground truth that real EEG does not.

---

# 11. What the project currently supports

The experiments collectively support a progression:

```text
Simple inverse problem
        ↓
Independent Gaussian noise
        ↓
No covariance advantage
        ↓
Structured covariance introduced
        ↓
Covariance-aware localization improves
        ↓
Real EEG residuals analysed
        ↓
Reproducible spatial covariance observed
        ↓
Empirical covariance transferred to simulator
        ↓
Covariance-aware localization improves
```

This progression is intentional.

Each stage answers a different question.

---

# 12. What this project does NOT prove

This is an important part of the project.

### It does not prove that EEG noise is definitely structured.

The real EEG residuals contain more than instrument noise.

### It does not prove that EEG correlation equals noise correlation.

Correlation is only evidence of statistical dependence.

### It does not demonstrate source localization accuracy on the real MNE dataset.

The real dataset has no known source ground truth for this experiment.

### It does not establish a new covariance-aware localization method.

Covariance-aware statistical modelling already exists in inverse problems.

### It does not establish universal improvement.

The simulator results depend on geometry, source configuration, electrode placement, noise scale, covariance structure, and other modelling choices.

The results should therefore be interpreted as evidence under the tested conditions, not as a universal claim.

---

# 13. Project structure

```text
eeg-structured-noise-lab/
│
├── experiments/
│   ├── 01_independent_noise_control.py
│   ├── 02_structured_covariance.py
│   ├── 03_real_eeg_covariance_reproducibility.py
│   └── 04_real_eeg_covariance_bridge.py
│
├── prototypes/
│   ├── v0_toy_2d_forward_inverse.py
│   ├── v1_2d_gaussian_noise.py
│   ├── v1_5_2d_off_grid_source.py
│   ├── v2_2d_dipole_orientation.py
│   └── v3_3d_sphere_dipole.py
│
├── real_eeg/
│   └── 01_event_locked_residual_correlation.py
│
├── src/
│   ├── geometry.py
│   ├── forward.py
│   ├── localization.py
│   └── noise.py
│
├── figures/
├── docs/
├── report/
└── scripts/
```

The repository intentionally contains the **curated research code**, not the original full laboratory workspace.

The original EEG/MNE dataset is not stored in the repository.

Large EEG data files such as `.fif` and `.edf` are excluded through `.gitignore`.

---

# 14. Reproducibility

The main experiments are written as executable Python modules.

From the project root:

```bash
python -m experiments.01_independent_noise_control
python -m experiments.02_structured_covariance
python -m experiments.03_real_eeg_covariance_reproducibility
python -m experiments.04_real_eeg_covariance_bridge
```

The simulator components are separated into reusable modules:

```text
geometry.py
    ↓
forward.py
    ↓
noise.py
    ↓
localization.py
```

This separation keeps the physical model, uncertainty model, and inverse procedure conceptually distinct.

---

# 15. Why I built it this way

This project deliberately avoids starting with a large black-box machine-learning model.

The progression is:

> **understand the physics → build the forward model → build the inverse model → add uncertainty → model covariance → test controlled localization → inspect real EEG → connect the two**

The point is not to hide the difficult parts behind a library call.

The point is to understand what each assumption is doing.

That is also why the repository preserves the earlier prototypes.

The 2D models are not obsolete code.

They document how the final 3D experiment was built from simpler pieces.

---

# 16. Current status

### Completed

- First-principles 2D forward/inverse model
- Gaussian measurement-noise control
- Off-grid source experiment
- Dipole orientation model
- 3D spherical simulator
- Independent-noise control experiment
- Structured covariance experiment
- Multi-seed covariance evaluation
- Event-locked real EEG residual analysis
- Real-vs-shuffled covariance reproducibility analysis
- Empirical real-EEG covariance bridge
- Shared reusable simulation modules
- Curated public GitHub repository

### Next scientific questions

The current results motivate further investigation rather than closing the question.

Potential next steps include:

- testing additional covariance structures
- varying electrode geometry
- testing different source locations and orientations
- separating spatial and temporal covariance effects
- investigating covariance estimation uncertainty
- evaluating robustness to covariance mismatch
- testing whether the advantage survives more realistic forward models
- comparing against established inverse methods

These should be treated as future experiments, not assumptions about the current result.

---

# 17. The central idea in one sentence

> **This project investigates whether learning and explicitly modelling reproducible structure in EEG measurement variability can make an inverse source-localization problem more statistically informed than treating all measurement uncertainty as independent noise.**

---

## Author

Built as a first-principles computational research project exploring the intersection of:

- EEG
- inverse problems
- statistical signal modelling
- covariance estimation
- computational neuroscience
- physics-inspired modelling
- numerical simulation

The project is intentionally experimental and transparent: every major result is built from simpler models that can be inspected independently.





