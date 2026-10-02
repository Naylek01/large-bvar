# Large Bayesian VARs — Chan (2020) replication

Python replication and extension of the empirical forecasting exercise in **Chan (2020), _Large Bayesian Vector Autoregressions_**.

The project implements the seven large BVAR specifications in a common 20-variable VAR(4) framework, keeps the original real-time forecasting logic, and adds a compact Python interface for posterior estimation, recursive forecasts, MCMC diagnostics, and structural analysis.

## Project structure

```text
large-bvar/
├── notebooks/
│   └── 01_large_bvar.ipynb
├── src/
│   ├── bvar_model_function.py
│   └── bvar_plot_function.py
├── data/
│   └── chan_2020/
├── results/
│   └── recursive/              # per-model recursive checkpoints
├── requirements.txt
└── README.md
```

The notebook is deliberately kept as the presentation layer. Estimation, forecasting, diagnostics, structural calculations, and recursive-comparison logic live in the two Python modules.

## Seven models

All models use the same 20-variable VAR(4), but differ in their prior/covariance treatment and in the dynamics of the reduced-form innovations.

| Model | Prior branch | Common SV | Student-t | MA(1) |
|---|---|:---:|:---:|:---:|
| `BVAR-Minn` | Minnesota, fixed diagonal covariance | No | No | No |
| `BVAR-NCP` | Natural-conjugate | No | No | No |
| `BVAR-IP` | Independent Normal + inverse-Wishart | No | No | No |
| `BVAR-SSVS` | SSVS + inverse-Wishart | No | No | No |
| `BVAR-CSV` | Natural-conjugate | Yes | No | No |
| `BVAR-CSV-t` | Natural-conjugate | Yes | Yes | No |
| `BVAR-CSV-t-MA` | Natural-conjugate | Yes | Yes | Yes |

The NCP error-dynamics branch is nested as

$$
\text{BVAR-NCP}
\longrightarrow
\text{BVAR-CSV}
\longrightarrow
\text{BVAR-CSV-t}
\longrightarrow
\text{BVAR-CSV-t-MA}.
$$

Minnesota, IP, and SSVS are separate prior/covariance branches; they are not obtained by switching off CSV, Student-t, or MA states.

## Econometric specification

### Reduced-form VAR

The common reduced-form model is

$$
y_t = c + B_1 y_{t-1} + \cdots + B_p y_{t-p} + u_t,
\qquad p=4.
$$

With 20 variables, each equation has

$$
k = 1 + np = 1 + 20\times 4 = 81
$$

regressors, for a total of

$$
20\times 81 = 1620
$$

VAR coefficients.

### Minnesota prior

For `BVAR-Minn`, the coefficient prior follows the Minnesota construction used in `prior_Minn.m`. Prior means are zero, the residual covariance matrix is fixed and diagonal, and lag variances shrink with the lag order.

For an own lag at lag $\ell$,

$$
\operatorname{Var}(B_{ii,\ell}) = \frac{c_1}{\ell^2}.
$$

For a cross lag,

$$
\operatorname{Var}(B_{ij,\ell})
=
\frac{c_2\,\sigma_i^2}{\ell^2\sigma_j^2},
\qquad i\neq j.
$$

The intercept variance is $c_3$. The Chan settings used here are

$$
c_1=0.2^2,\qquad c_2=0.1^2,\qquad c_3=100.
$$

### Natural-conjugate prior

`BVAR-NCP` and the CSV / Student-t / MA extensions use

$$
A\mid\Sigma \sim MN(A_0,V_A,\Sigma),
$$

$$
\Sigma \sim IW(\nu_0,S_0).
$$

The notebook reproduces the hyperparameter construction in `prior_NC.m`, including $A_0=0$, lag shrinkage based on univariate AR residual variances, and $\nu_0=n+3$.

### Independent prior

`BVAR-IP` separates the coefficient prior from the residual covariance prior:

$$
\beta \sim N(\beta_0,V_\beta),
$$

$$
\Sigma \sim IW(\nu_0,S_0).
$$

Because the coefficient prior is independent of $\Sigma$, the model is sampled with Gibbs steps rather than the natural-conjugate direct draw.

### SSVS prior

`BVAR-SSVS` adds one latent inclusion indicator to each coefficient. Conditional on $\gamma_j$,

$$
\beta_j\mid \gamma_j=0 \sim N\!\left(0,V_j^{\text{Minn}}\right),
$$

$$
\beta_j\mid \gamma_j=1 \sim N(0,\kappa_1),
$$

with

$$
\gamma_j\sim \operatorname{Bernoulli}(q),
\qquad q=0.5,
\qquad \kappa_1=10.
$$

The notebook retains `gamma_share`, the fraction of coefficients in the less-shrunk slab state.

## Common stochastic volatility, Student-t errors, and MA(1)

The richest specification, `BVAR-CSV-t-MA`, writes the reduced-form innovation as

$$
u_t = e_t + \psi e_{t-1}.
$$

Conditional on the common volatility state and Student-t scale mixture,

$$
e_t\mid h_t,\lambda_t,\Sigma
\sim
N\!\left(0,\lambda_t e^{h_t}\Sigma\right).
$$

The common stochastic-volatility state follows

$$
h_t = \rho h_{t-1}+\eta_t,
$$

$$
\eta_t\sim N(0,\sigma_h^2).
$$

Student-t innovations are represented through

$$
\lambda_t\sim IG\!\left(\frac{\nu}{2},\frac{\nu}{2}\right).
$$

The generalized engine switches these components off literally:

- `stochastic_volatility=False`: $h_t=0$, with $\rho$ and $\sigma_h^2$ inactive;
- `student_t=False`: $\lambda_t=1$, with $\nu$ inactive;
- `ma1=False`: $\psi=0$ and the MA update is skipped.

## Estimation and MCMC diagnostics

The notebook estimates every model on all 20 variables at the final origin using

```python
NSIMS = 5000
BURNIN = 100
SEED = 0
```

`BVAR-Minn` and `BVAR-NCP` use direct posterior draws. `BVAR-IP`, `BVAR-SSVS`, `BVAR-CSV`, `BVAR-CSV-t`, and `BVAR-CSV-t-MA` require Gibbs and/or MH updates.

For MCMC specifications, the diagnostics include:

- trace plots for representative coefficients and active latent parameters;
- autocorrelation functions;
- effective sample size (ESS);
- Monte Carlo standard error (MCSE);
- first-order autocorrelation;
- MH / acceptance-rejection shares when applicable.

For IP and SSVS, the joint coefficient conditional is the same Gaussian conditional as in the MATLAB SUR formulation. The Python implementation solves the joint precision system with block-preconditioned CG and a MINRES fallback rather than MATLAB sparse Cholesky; this is a numerical implementation difference, not a change in the target posterior.

## Forecasting

### One fitted origin

The public interface for posterior forecasting is

```python
forecast = forecast_bvar(
    fit,
    sample,
    horizon=12,
    seed=SEED,
    t=T,
    T=T,
)
```

The function reuses the posterior draws already stored by `fit_bvar`; it does **not** re-estimate the model.

For every posterior draw, the forecast engine recursively simulates future observations and feeds simulated values back into the lag vector. The predictive density is aggregated over posterior draws with the log-mean-exp identity:

$$
\log p(y_{T+h}\mid Y_T)
=
\log\left[
\frac{1}{S}
\sum_{s=1}^{S}
p(y_{T+h}\mid\theta^{(s)},Y_T)
\right].
$$

For the MA(1) specification, the innovation state is propagated recursively. The same MA filter is explicitly inverted in the historical decomposition:

$$
e_t = u_t-\psi e_{t-1}.
$$

## Structural analysis

Each model can be analysed with the same structural pipeline:

- sign-restriction diagnostics;
- impulse-response functions (IRFs);
- forecast-error variance decompositions (FEVDs);
- historical decompositions (HDs).

The default monetary-policy sign restrictions are

| Variable | Sign |
|---|:---:|
| Federal funds rate | $+$ |
| Real GDP | $-$ |
| PCE | $-$ |

The implementation uses base-$\Sigma$ shock units and keeps the MA(1) term in the IRF recursion when it is active. Historical decompositions include a reconstruction check; numerical reconstruction errors should be close to machine precision.

The notebook also reports a geometric sign-identification diagnostic based on the acceptance volume and angular dispersion of admissible shock directions.

## Full Chan recursive forecasting comparison

The final section re-estimates **all seven models** over the historical real-time vintages and compares their nowcasts and one-quarter-ahead forecasts.

The evaluation focuses on the four core variables used in the replication:

- Real GDP;
- Industrial production;
- Unemployment rate;
- PCE inflation.

Point forecasts are compared with RMSFE,

$$
\operatorname{RMSFE}
=
\sqrt{
\frac{1}{M}
\sum_{m=1}^{M}
\left(y_m-\widehat y_m\right)^2
},
$$

while density forecasts are compared with average log predictive likelihood,

$$
\operatorname{ALPL}
=
\frac{1}{M}
\sum_{m=1}^{M}
\log p(y_m\mid\mathcal I_{m-1}).
$$

Lower RMSFE is better; higher ALPL is better.

The recursive block is enabled by default:

```python
RUN_FULL_REPLICATION = True
```

To keep the historical seven-model exercise tractable, it uses a separate Monte Carlo budget:

```python
REPLICATION_NSIMS = 500
REPLICATION_BURNIN = 100
```

The final-origin model sections remain at 5000 retained draws. To reproduce the original Monte Carlo budget in the full historical loop as closely as possible, set `REPLICATION_NSIMS = 5000`.

Each completed recursive model is checkpointed under

```text
results/recursive/
```

so an interrupted run can resume without recomputing completed models. Set `REPLICATION_FORCE=True` only when the checkpoints should be ignored and every model rerun.

## Numerical performance

The IP and SSVS samplers perform many relatively small linear-algebra operations. On the development machine, unrestricted OpenBLAS threading was substantially slower than a small thread pool. The notebook therefore fixes BLAS to two threads:

```python
from threadpoolctl import threadpool_limits

BLAS_THREADS = 2
_BLAS_LIMITER = threadpool_limits(
    limits=BLAS_THREADS,
    user_api="blas",
)
```

This setting changes only computational scheduling; it does not change the model, posterior, random seed, or forecasting equations. If the project is moved to a different machine, benchmark `BLAS_THREADS` before increasing it.

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then start JupyterLab:

```powershell
jupyter lab
```

Open:

```text
notebooks/01_large_bvar.ipynb
```

## Export the executed notebook to HTML

If the notebook has already been executed and the outputs should simply be preserved:

```powershell
cd notebooks
jupyter nbconvert --to html 01_large_bvar.ipynb --output 01_large_bvar_results.html
```

This converts the stored notebook outputs without rerunning the seven-model recursive exercise.

## Data

The loader follows the data contract and spreadsheet ranges used by `main_forecasting.m`. Place the Chan replication data under

```text
data/chan_2020/
```

The exact source files, ranges, transformations, and vintage structures can be inspected in the notebook through `data_source_table()`.

## Source fidelity and deliberate implementation choices

The project is designed to remain close to the original MATLAB code:

- same seven model specifications;
- same VAR dimension and lag order;
- same prior branches and main hyperparameters;
- same real-time vintage logic;
- same recursive nowcast / one-quarter-ahead design;
- same Student-t, common-SV, and MA(1) state logic;
- MA(1) retained in recursive forecasts, IRFs, and historical decompositions.

Two practical Python choices are explicit:

1. IP/SSVS use an equivalent iterative linear solve instead of MATLAB sparse Cholesky;
2. the default full historical comparison uses 500 retained draws per origin, while the final-origin estimates use 5000. This can be changed back to 5000 for the historical loop.

## Reference

If you use the original replication material, cite:

> Chan, J. C. C. (2020). **Large Bayesian Vector Autoregressions.** In P. Fuleky (Ed.), _Macroeconomic Forecasting in the Era of Big Data_, pp. 95–125. Springer, Cham.

The original MATLAB README states that the replication code is free to use for academic purposes provided that the paper is cited.
