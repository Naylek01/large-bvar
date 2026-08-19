"""Generalized Chan (2020) large-BVAR replication engine.

The module keeps the project intentionally compact: one econometric engine plus
one plotting module.  The public API exposes the seven large BVARs in Chan
(2020) as presets of the same framework:

    BVAR-Minn
    BVAR-NCP
    BVAR-IP
    BVAR-SSVS
    BVAR-CSV
    BVAR-CSV-t
    BVAR-CSV-t-MA

The NCP -> CSV -> CSV-t -> CSV-t-MA branch is nested through three switches:
``stochastic_volatility``, ``student_t`` and ``ma1``.  Minnesota, IP and SSVS
remain distinct prior/covariance branches, exactly as in the supplied MATLAB
code; they are not obtained by setting CSV/t/MA parameters to zero.

The translation follows the supplied MATLAB files and preserves their data
contract, prior hyperparameters, recursive evaluation windows and forecasting
logic.  NumPy/SciPy random-number generators differ from MATLAB, so numerical
replication should be assessed by posterior/forecast results rather than by
bitwise-identical draws.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special, stats
from scipy.sparse import linalg as sparse_linalg


# =============================================================================
# CHAN (2020) DATA CONTRACT
# =============================================================================

CORE_VARIABLES = [0, 6, 7, 11]
CORE_LABELS = [
    "GDP growth",
    "Industrial production growth",
    "Unemployment rate",
    "PCE inflation",
]

TCODE = np.array(
    [5, 5, 5, 5, 1, 5, 5, 1, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 5],
    dtype=int,
)

VAR_TYPE = np.array(
    [1, 1, 1, 1, 1, 1, 3, 2, 3, 3, 3, 1, 1, 4, 4, 4, 4, 4, 4, 4],
    dtype=int,
)

SOURCE_SPECS = {
    1: ("ROUTPUTQvQd.xlsx", "AI70:HA283", "rt"),
    2: ("RCONQvQd.xlsx", "AI70:HA283", "rt"),
    3: ("rinvbfQvQd.xlsx", "AI70:HA283", "rt"),
    4: ("rinvresidQvQd.xlsx", "AI70:HA283", "rt"),
    5: ("RNXQvQd.xlsx", "AI71:HA283", "rt"),
    6: ("npiQvQd.xlsx", "AI70:HA283", "rt"),
    7: ("iptMvMd.xlsx", "EF542:YI1184", "rt"),
    8: ("rucQvMd.xlsx", "AI209:HA848", "rt"),
    9: ("employMvMd.xlsx", "DG302:XJ944", "rt"),
    10: ("hMvMd.xlsx", "AD206:UG848", "rt"),
    11: ("hstartsMvMd.xlsx", "BW206:VX848", "rt"),
    12: ("pconQvQd.xlsx", "AI70:HA283", "rt"),
    13: ("pimpQvQd.xlsx", "AI70:HA283", "rt"),
    14: ("FEDFUNDS.xls", "B129:B770", "nonrev"),
    15: ("GS1.xls", "B144:B785", "nonrev"),
    16: ("GS10.xls", "B144:B785", "nonrev"),
    17: ("BAAFFM.xls", "B129:B770", "nonrev"),
    18: ("ISM-MAN_PMI.csv", "B196:B837", "nonrev"),
    19: ("ISM-MAN_NEWORDERS.xls", "B196:B837", "nonrev"),
    20: ("SP500.xlsx", "B62:B705", "nonrev"),
}

FULL_VARIABLE_LABELS = [
    "Real GDP",
    "Real consumption",
    "Business fixed investment",
    "Residential investment",
    "Net exports",
    "Nonfarm payrolls / income proxy",
    "Industrial production",
    "Unemployment rate",
    "Employment",
    "Hours",
    "Housing starts",
    "PCE",
    "Import prices",
    "Federal funds rate",
    "1-year Treasury yield",
    "10-year Treasury yield",
    "BAA spread",
    "ISM PMI",
    "ISM new orders",
    "S&P 500",
]


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class BVARConfig:
    """Configuration of one Chan-style BVAR.

    Parameters controlling the error dynamics are deliberately booleans.  The
    latent variables remain internal to the sampler:

    * stochastic_volatility=False  -> h_t is fixed at zero;
    * student_t=False              -> lambda_t is fixed at one and Gaussian
                                      predictive densities are used;
    * ma1=False                    -> psi is fixed at zero.
    """

    name: str = "BVAR-CSV-t-MA"
    prior: str = "ncp"  # minnesota, ncp, ip, ssvs
    stochastic_volatility: bool = True
    student_t: bool = True
    ma1: bool = True

    # VAR / shrinkage hyperparameters.  Values match main_forecasting.m.
    p: int = 4
    c1: float = 0.2**2
    c2: float = 100.0
    c3: float = 100.0

    # SSVS.
    q: float = 0.5
    kappa1: float = 10.0

    # Common stochastic volatility.
    rho0: float = 0.9
    Vrho: float = 0.2**2
    nuh0: float = 5.0
    Sh0: float = 0.01 * (5.0 - 1.0)
    rho_init: float = 0.8
    sigh2_init: float = 0.1

    # Student-t degrees of freedom.
    nu_init: float = 5.0
    nu_upper: float = 50.0

    # MA(1).
    psi0: float = 0.0
    Vpsi: float = 1.0
    psi_init: float = 0.1
    psi_bound: float = 0.99
    psi_hessian_every: int = 100


MODEL_PRESETS: Dict[str, BVARConfig] = {
    "BVAR-Minn": BVARConfig(
        name="BVAR-Minn",
        prior="minnesota",
        stochastic_volatility=False,
        student_t=False,
        ma1=False,
        c1=0.2**2,
        c2=0.1**2,
        c3=100.0,
    ),
    "BVAR-NCP": BVARConfig(
        name="BVAR-NCP",
        prior="ncp",
        stochastic_volatility=False,
        student_t=False,
        ma1=False,
        c1=0.2**2,
        c2=100.0,
    ),
    "BVAR-IP": BVARConfig(
        name="BVAR-IP",
        prior="ip",
        stochastic_volatility=False,
        student_t=False,
        ma1=False,
        c1=0.2**2,
        c2=0.1**2,
        c3=100.0,
    ),
    "BVAR-SSVS": BVARConfig(
        name="BVAR-SSVS",
        prior="ssvs",
        stochastic_volatility=False,
        student_t=False,
        ma1=False,
        c1=0.2**2,
        c2=0.1**2,
        c3=100.0,
        q=0.5,
        kappa1=10.0,
    ),
    "BVAR-CSV": BVARConfig(
        name="BVAR-CSV",
        prior="ncp",
        stochastic_volatility=True,
        student_t=False,
        ma1=False,
        c1=0.2**2,
        c2=100.0,
    ),
    "BVAR-CSV-t": BVARConfig(
        name="BVAR-CSV-t",
        prior="ncp",
        stochastic_volatility=True,
        student_t=True,
        ma1=False,
        c1=0.2**2,
        c2=100.0,
    ),
    "BVAR-CSV-t-MA": BVARConfig(
        name="BVAR-CSV-t-MA",
        prior="ncp",
        stochastic_volatility=True,
        student_t=True,
        ma1=True,
        c1=0.2**2,
        c2=100.0,
    ),
}


def get_model_config(model: str | BVARConfig = "BVAR-CSV-t-MA", **overrides: Any) -> BVARConfig:
    """Return a validated configuration, optionally overriding preset fields."""
    if isinstance(model, BVARConfig):
        config = model
    else:
        if model not in MODEL_PRESETS:
            choices = ", ".join(MODEL_PRESETS)
            raise ValueError(f"Unknown model {model!r}. Choose one of: {choices}")
        config = MODEL_PRESETS[model]

    if overrides:
        config = replace(config, **overrides)

    validate_config(config)
    return config


def validate_config(config: BVARConfig) -> None:
    """Validate combinations implemented by the source-faithful engine."""
    if config.prior not in {"minnesota", "ncp", "ip", "ssvs"}:
        raise ValueError("prior must be 'minnesota', 'ncp', 'ip', or 'ssvs'")
    if config.p < 1:
        raise ValueError("p must be at least 1")
    if config.prior != "ncp" and (
        config.stochastic_volatility or config.student_t or config.ma1
    ):
        raise ValueError(
            "Chan's CSV/t/MA specifications use the natural-conjugate prior. "
            "For source-faithful replication, dynamic-error switches are only "
            "available with prior='ncp'."
        )
    if config.student_t and not config.stochastic_volatility:
        raise ValueError(
            "The seven Chan presets contain Student-t errors only in the CSV branch."
        )
    if config.ma1 and not (config.stochastic_volatility and config.student_t):
        raise ValueError(
            "In Chan's seven-model comparison, MA(1) appears only in BVAR-CSV-t-MA."
        )
    if not 0 < config.q < 1:
        raise ValueError("q must lie strictly between 0 and 1")
    if not 0 < config.psi_bound < 1:
        raise ValueError("psi_bound must lie strictly between 0 and 1")


def model_presets_table() -> pd.DataFrame:
    """Readable map from the seven model names to the generalized switches."""
    rows = []
    for name, cfg in MODEL_PRESETS.items():
        rows.append(
            {
                "model": name,
                "prior": cfg.prior,
                "CSV": cfg.stochastic_volatility,
                "Student-t": cfg.student_t,
                "MA(1)": cfg.ma1,
                "c1": cfg.c1,
                "c2": cfg.c2,
                "c3": cfg.c3 if cfg.prior != "ncp" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def nested_ncp_configs() -> Dict[str, BVARConfig]:
    """The exact NCP -> CSV -> CSV-t -> CSV-t-MA nesting used in the project."""
    return {name: MODEL_PRESETS[name] for name in (
        "BVAR-NCP", "BVAR-CSV", "BVAR-CSV-t", "BVAR-CSV-t-MA"
    )}


# =============================================================================
# DATA LOADING
# =============================================================================

def _excel_col(col: str) -> int:
    value = 0
    for c in col.upper():
        value = value * 26 + ord(c) - 64
    return value - 1


def _read_range(path: str | Path, cell_range: str) -> np.ndarray:
    """Read the exact spreadsheet range used in main_forecasting.m."""
    match = re.match(r"([A-Z]+)(\d+):([A-Z]+)(\d+)", cell_range.upper())
    if match is None:
        raise ValueError(f"Invalid range: {cell_range}")

    c0, r0, c1, r1 = match.groups()
    r0, r1 = int(r0) - 1, int(r1) - 1
    c0, c1 = _excel_col(c0), _excel_col(c1)

    kwargs = dict(
        header=None,
        skiprows=r0,
        nrows=r1 - r0 + 1,
        usecols=list(range(c0, c1 + 1)),
    )

    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, **kwargs)
    else:
        frame = pd.read_excel(path, **kwargs)

    return frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def load_chan_data(data_dir: str | Path):
    """Load the 20 data files with the exact ranges from main_forecasting.m."""
    data_dir = Path(data_dir)
    rt_data: Dict[str, np.ndarray] = {}
    nonrev_data: Dict[str, np.ndarray] = {}

    for i, (filename, cell_range, kind) in SOURCE_SPECS.items():
        values = _read_range(data_dir / filename, cell_range)
        if kind == "rt":
            rt_data[f"var{i}"] = values
        else:
            nonrev_data[f"var{i}"] = values.reshape(-1)

    return rt_data, nonrev_data, TCODE.copy(), VAR_TYPE.copy()


def data_source_table(tcode: np.ndarray = TCODE, var_type: np.ndarray = VAR_TYPE) -> pd.DataFrame:
    """Describe the 20 source series, transformations and vintage structures."""
    rows = []
    for i, (filename, cell_range, kind) in SOURCE_SPECS.items():
        rows.append(
            {
                "variable": i,
                "label": FULL_VARIABLE_LABELS[i - 1],
                "file": filename,
                "range": cell_range,
                "source": kind,
                "transform": "400*dlog" if int(tcode[i - 1]) == 5 else "level",
                "data structure": {
                    1: "Q vintage / Q observations",
                    2: "Q vintage / M observations",
                    3: "M vintage / M observations",
                    4: "non-revised / M observations",
                }[int(var_type[i - 1])],
            }
        )
    return pd.DataFrame(rows)


def _monthly_to_quarterly(values: np.ndarray) -> np.ndarray:
    """Equivalent to MATLAB mean(reshape(x,3,n_q))'."""
    values = np.asarray(values, dtype=float).reshape(-1)
    n_q = len(values) // 3
    return values[: 3 * n_q].reshape(n_q, 3).mean(axis=1)


def loaddata(
    rt_data: Mapping[str, np.ndarray],
    nonrev_data: Mapping[str, np.ndarray],
    t: int,
    T0: int,
    tcode: np.ndarray,
    var_type: np.ndarray,
):
    """Direct Python translation of the supplied loaddata.m."""
    n = len(var_type)
    data_t = np.zeros((t - 2, n))
    data_tpk = np.zeros((5, n))

    for i in range(n):
        key = f"var{i + 1}"

        if var_type[i] == 1:
            all_vintages = rt_data[key]
            values = all_vintages[:, t - T0]
            final_values = all_vintages[:, -1]
        elif var_type[i] == 2:
            all_vintages = rt_data[key]
            values = _monthly_to_quarterly(all_vintages[:, t - T0])
            final_values = _monthly_to_quarterly(all_vintages[:, -1])
        elif var_type[i] == 3:
            all_vintages = rt_data[key]
            vintage = (t - T0) * 3 + 2
            values = _monthly_to_quarterly(all_vintages[:, vintage])
            final_values = _monthly_to_quarterly(all_vintages[:, -1])
        elif var_type[i] == 4:
            values = _monthly_to_quarterly(nonrev_data[key])
            final_values = values
        else:
            raise ValueError(f"Unknown var_type: {var_type[i]}")

        if tcode[i] == 5:
            data_t[:, i] = 400 * np.log(values[1:t - 1] / values[:t - 2])
            final_transformed = 400 * np.log(final_values[1:] / final_values[:-1])
            # MATLAB: y_last_vin(t-1:t+3), converted from 1-based indexing.
            data_tpk[:, i] = final_transformed[t - 2:t + 3]
        elif tcode[i] == 1:
            data_t[:, i] = values[:t - 2]
            data_tpk[:, i] = final_values[t - 2:t + 3]
        else:
            raise ValueError(f"Unknown tcode: {tcode[i]}")

    return data_t, data_tpk


def trim_data(data_t: np.ndarray):
    """Apply the missing-data trimming in main_forecasting.m."""
    data_t = np.asarray(data_t, dtype=float)
    is_last_miss = np.isnan(data_t[-1]).any()
    if is_last_miss:
        data_t = data_t[:-1]

    missing_rows = np.where(np.isnan(data_t).any(axis=1))[0]
    if len(missing_rows):
        data_t = data_t[missing_rows[-1] + 1:]

    return data_t, bool(is_last_miss)


def prepare_large_bvar_sample(
    rt_data,
    nonrev_data,
    tcode,
    var_type,
    t: int = 208,
    T0: int = 41,
    p: int = 4,
):
    """Build the 20-variable sample at one recursive forecast origin."""
    data_t, data_tpk = loaddata(rt_data, nonrev_data, t, T0, tcode, var_type)
    data_t, is_last_miss = trim_data(data_t)
    if len(data_t) <= p:
        raise ValueError("Not enough complete observations after trimming.")

    return {
        "data": data_t,
        "Y0": data_t[:p],
        "shortYt": data_t[p:],
        "targets": data_tpk,
        "is_last_miss": is_last_miss,
        "t": t,
        "p": p,
        "variables": FULL_VARIABLE_LABELS.copy(),
    }


def prepare_bvar_small_sample(
    rt_data,
    nonrev_data,
    tcode,
    var_type,
    t: int = 208,
    T0: int = 41,
    p: int = 4,
):
    """Legacy four-variable BVAR-small sample retained for compatibility."""
    large = prepare_large_bvar_sample(rt_data, nonrev_data, tcode, var_type, t, T0, p)
    return {
        "data": large["data"][:, CORE_VARIABLES],
        "Y0": large["Y0"][:, CORE_VARIABLES],
        "shortYt": large["shortYt"][:, CORE_VARIABLES],
        "targets": large["targets"][:, CORE_VARIABLES],
        "is_last_miss": large["is_last_miss"],
        "t": t,
        "p": p,
        "variables": CORE_LABELS.copy(),
    }


# =============================================================================
# VAR DESIGN AND OLS
# =============================================================================

def build_var_design(Y0: np.ndarray, shortYt: np.ndarray, p: int = 4) -> np.ndarray:
    """Construct [1, y_{t-1}', ..., y_{t-p}'] in Chan's regressor order."""
    Y0 = np.asarray(Y0, dtype=float)
    shortYt = np.asarray(shortYt, dtype=float)
    Tt, n = shortYt.shape
    tmpY = np.vstack([Y0[-p:], shortYt])
    Z = np.ones((Tt, 1 + n * p))
    for lag in range(1, p + 1):
        Z[:, 1 + (lag - 1) * n:1 + lag * n] = tmpY[p - lag:p - lag + Tt]
    return Z


def coefficient_names(variable_names: Sequence[str], p: int = 4) -> list[str]:
    return ["const"] + [
        f"{name} L{lag}"
        for lag in range(1, p + 1)
        for name in variable_names
    ]


def ols_var(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    p: int = 4,
    variable_names: Optional[Sequence[str]] = None,
):
    """OLS benchmark using the same VAR design matrix as every BVAR branch."""
    shortYt = np.asarray(shortYt, dtype=float)
    n = shortYt.shape[1]
    names = list(variable_names or [f"var{i+1}" for i in range(n)])
    Z = build_var_design(Y0, shortYt, p=p)
    B_ols = np.linalg.solve(Z.T @ Z, Z.T @ shortYt)
    residuals = shortYt - Z @ B_ols
    Sigma_ols = residuals.T @ residuals / len(shortYt)
    return {
        "B": B_ols,
        "residuals": residuals,
        "Sigma": Sigma_ols,
        "B_table": pd.DataFrame(B_ols, index=coefficient_names(names, p), columns=names),
        "Sigma_table": pd.DataFrame(Sigma_ols, index=names, columns=names),
    }


def ols_bvar_small(Y0, shortYt, p: int = 4, variable_names=CORE_LABELS):
    """Backward-compatible alias used by the original notebook."""
    return ols_var(Y0, shortYt, p=p, variable_names=variable_names)


# =============================================================================
# PRIORS
# =============================================================================

def ar_residual_variances(Y0: np.ndarray, shortYt: np.ndarray, p: int = 4) -> np.ndarray:
    """AR(p) residual variances used by prior_Minn.m and prior_NC.m."""
    Tt, n = shortYt.shape
    tmpY = np.vstack([Y0[-p:], shortYt])
    sig2 = np.zeros(n)
    for i in range(n):
        Zi = np.ones((Tt, p + 1))
        for lag in range(1, p + 1):
            Zi[:, lag] = tmpY[p - lag:p - lag + Tt, i]
        y = tmpY[p:p + Tt, i]
        beta = np.linalg.solve(Zi.T @ Zi, Zi.T @ y)
        sig2[i] = np.mean((y - Zi @ beta) ** 2)
    return sig2


def prior_minnesota(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    p: int = 4,
    c1: float = 0.2**2,
    c2: float = 0.1**2,
    c3: float = 100.0,
):
    """Source-faithful translation of prior_Minn.m."""
    _, n = shortYt.shape
    k = 1 + n * p
    beta_minn = np.zeros(k * n)
    V_minn = np.zeros(k * n)
    sig2 = ar_residual_variances(Y0, shortYt, p=p)

    count = 0
    for i in range(n):
        for ii in range(k):
            if ii == 0:
                V_minn[count] = c3
            else:
                j = (ii - 1) % n
                lag = (ii - 1) // n + 1
                if i == j:
                    V_minn[count] = c1 / lag**2
                else:
                    V_minn[count] = c2 * sig2[i] / (lag**2 * sig2[j])
            count += 1

    return {
        "type": "minnesota",
        "mean": beta_minn,
        "variance": V_minn,
        "sig2": sig2,
        "p": p,
        "n": n,
        "k": k,
        "c1": c1,
        "c2": c2,
        "c3": c3,
    }


def prior_natural_conjugate(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    p: int = 4,
    c1: float = 0.2**2,
    c2: float = 100.0,
):
    """Source-faithful translation of prior_NC.m."""
    _, n = shortYt.shape
    k = 1 + n * p
    sig2 = ar_residual_variances(Y0, shortYt, p=p)
    A0 = np.zeros((k, n))
    VA0 = np.zeros(k)
    VA0[0] = c2
    for ii in range(1, k):
        j = (ii - 1) % n
        lag = (ii - 1) // n + 1
        VA0[ii] = c1 / (lag**2 * sig2[j])
    S0 = np.diag(sig2)
    nu0 = n + 3
    return {
        "type": "ncp",
        "A0": A0,
        "VA0": VA0,
        "S0": S0,
        "nu0": nu0,
        "sig2": sig2,
        "p": p,
        "n": n,
        "k": k,
        "c1": c1,
        "c2": c2,
    }


def prior_table(prior: Mapping[str, Any], variable_names: Sequence[str]) -> pd.DataFrame:
    """Readable prior table for Minnesota or natural-conjugate priors."""
    n = int(prior["n"])
    k = int(prior["k"])
    p = int(prior["p"])
    coef_names = coefficient_names(variable_names, p)
    rows = []

    if prior["type"] == "minnesota":
        for i, equation in enumerate(variable_names):
            sl = slice(i * k, (i + 1) * k)
            for coef, mean, variance in zip(
                coef_names, prior["mean"][sl], prior["variance"][sl]
            ):
                rows.append({
                    "equation": equation,
                    "coefficient": coef,
                    "prior mean": mean,
                    "prior sd": np.sqrt(variance),
                })
    elif prior["type"] == "ncp":
        for equation in variable_names:
            for coef, variance_scale in zip(coef_names, prior["VA0"]):
                rows.append({
                    "equation": equation,
                    "coefficient": coef,
                    "prior mean": 0.0,
                    "VA0": variance_scale,
                })
    else:
        raise ValueError("Unsupported prior table type")

    return pd.DataFrame(rows)


# =============================================================================
# NUMERICAL HELPERS / LATENT-STATE SAMPLERS
# =============================================================================

def _safe_cholesky(A: np.ndarray, jitter: float = 1e-10, max_tries: int = 7) -> np.ndarray:
    """Cholesky with tiny diagonal jitter for floating-point roundoff only."""
    A = (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T) / 2
    eye = np.eye(A.shape[0])
    for attempt in range(max_tries):
        try:
            return np.linalg.cholesky(A + (jitter * 10**attempt) * eye)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky failed after jitter attempts")


def _draw_inverse_wishart(scale: np.ndarray, df: float, rng: np.random.Generator) -> np.ndarray:
    draw = stats.invwishart.rvs(df=df, scale=scale, random_state=rng)
    draw = np.asarray(draw, dtype=float)
    if draw.ndim == 0:
        draw = draw.reshape(1, 1)
    return (draw + draw.T) / 2


def _ma_inverse_filter(values: np.ndarray, psi: float) -> np.ndarray:
    """Solve (I + psi L) x = values without constructing the full matrix."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = values[t] - psi * out[t - 1]
    return out


def _whiten_rows(U: np.ndarray, CSig: np.ndarray) -> np.ndarray:
    """MATLAB U/CSig' using triangular solves."""
    return linalg.solve_triangular(CSig, U.T, lower=True).T


def _sv_prior_precision(Tt: int, rho: float, sigh2: float) -> np.ndarray:
    H = np.eye(Tt)
    if Tt > 1:
        H[np.arange(1, Tt), np.arange(Tt - 1)] = -rho
    diag = np.full(Tt, 1.0 / sigh2)
    diag[0] = (1.0 - rho**2) / sigh2
    return H.T @ (diag[:, None] * H)


def sample_h(
    s2: np.ndarray,
    rho: float,
    sigh2: float,
    h: np.ndarray,
    n: int,
    rng: np.random.Generator,
    tol: float = 1e-3,
    max_newton: int = 500,
    max_ar: int = 20000,
):
    """Translation of sample_h.m (mode-based accept-reject + MH correction)."""
    s2 = np.asarray(s2, dtype=float)
    h = np.asarray(h, dtype=float).copy()
    Tt = len(s2)
    HiSH = _sv_prior_precision(Tt, rho, sigh2)

    ht = h.copy()
    for _ in range(max_newton):
        exp_ht = np.exp(np.clip(ht, -700, 700))
        sieht = s2 / exp_ht
        fh = -n / 2 + 0.5 * sieht
        Gh = 0.5 * sieht
        Kh = HiSH + np.diag(Gh)
        newht = np.linalg.solve(Kh, fh + Gh * ht)
        if np.max(np.abs(newht - ht)) <= tol:
            ht = newht
            break
        ht = newht
    else:
        raise RuntimeError("sample_h Newton iteration did not converge")

    exp_ht = np.exp(np.clip(ht, -700, 700))
    sieht = s2 / exp_ht
    Gh = 0.5 * sieht
    Kh = HiSH + np.diag(Gh)
    CKh = _safe_cholesky(Kh)

    def log_kernel(x: np.ndarray) -> float:
        return float(
            -0.5 * x @ HiSH @ x
            - n / 2 * np.sum(x)
            - 0.5 * np.exp(np.clip(-x, -700, 700)) @ s2
        )

    logc = log_kernel(ht) + np.log(3.0)
    hc = None
    alpARc = None
    for _ in range(max_ar):
        z = rng.standard_normal(Tt)
        step = linalg.solve_triangular(CKh.T, z, lower=False)
        candidate = ht + step
        candidate_ar = (
            log_kernel(candidate)
            + 0.5 * (candidate - ht) @ Kh @ (candidate - ht)
            - logc
        )
        if candidate_ar > np.log(rng.random()):
            hc = candidate
            alpARc = float(candidate_ar)
            break
    if hc is None:
        raise RuntimeError("sample_h accept-reject step failed to accept")

    alpAR = float(log_kernel(h) + 0.5 * (h - ht) @ Kh @ (h - ht) - logc)
    if alpAR < 0:
        alpMH = 0.0
    elif alpARc < 0:
        alpMH = -alpAR
    else:
        alpMH = alpARc - alpAR

    accepted = alpMH > np.log(rng.random())
    return (hc if accepted else h), bool(accepted)


def sample_nu(
    lam: np.ndarray,
    nu: float,
    nu_upper: float,
    rng: np.random.Generator,
    tol: float = 1e-5,
    max_iter: int = 200,
):
    """Translation of sample_nu.m."""
    lam = np.asarray(lam, dtype=float)
    Tt = len(lam)
    sum1 = float(np.sum(np.log(lam)))
    sum2 = float(np.sum(1.0 / lam))

    def f_nu(x: float) -> float:
        return float(
            Tt * (x / 2 * np.log(x / 2) - special.gammaln(x / 2))
            - (x / 2 + 1) * sum1
            - x / 2 * sum2
        )

    def df_nu(x: float) -> float:
        return float(
            Tt / 2 * (np.log(x / 2) + 1 - special.digamma(x / 2))
            - 0.5 * (sum1 + sum2)
        )

    def d2f_nu(x: float) -> float:
        return float(Tt / (2 * x) - Tt / 4 * special.polygamma(1, x / 2))

    nut = float(nu)
    H_nu = d2f_nu(nut)
    for _ in range(max_iter):
        score = df_nu(nut)
        H_nu = d2f_nu(nut)
        if abs(score) <= tol:
            break
        nut = nut - score / H_nu
        if nut < 2:
            nut = 5.0
            H_nu = d2f_nu(nut)
            break

    Dnu = -1.0 / H_nu
    if not np.isfinite(Dnu) or Dnu <= 0:
        return float(nu), False

    nuc = nut + np.sqrt(Dnu) * rng.standard_normal()
    if not (2 < nuc < nu_upper):
        return float(nu), False

    lalp = (
        f_nu(nuc)
        - f_nu(nu)
        - 0.5 * (nu - nut) ** 2 / Dnu
        + 0.5 * (nuc - nut) ** 2 / Dnu
    )
    accepted = np.log(rng.random()) < min(0.0, lalp)
    return (float(nuc) if accepted else float(nu)), bool(accepted)


def llike_csv_ma(psi: float, U: np.ndarray, Sig: np.ndarray, h: np.ndarray) -> float:
    """Translation of llike_CSV_MA.m."""
    U = np.asarray(U, dtype=float)
    Tt, n = U.shape
    Utld = _ma_inverse_filter(U, psi)
    CSig = _safe_cholesky(Sig)
    c = (
        -Tt * n / 2 * np.log(2 * np.pi)
        - Tt * np.sum(np.log(np.diag(CSig)))
        - n / 2 * np.log(1 + psi**2)
    )
    tmp = _whiten_rows(Utld, CSig)
    s2 = np.sum(tmp**2, axis=1)
    return float(
        c
        - 0.5
        * (
            s2[0] / ((1 + psi**2) * np.exp(h[0]))
            + np.sum(s2[1:] / np.exp(h[1:]))
        )
    )


def _log_prior_psi(psi: float, config: BVARConfig) -> float:
    if abs(psi) > config.psi_bound:
        return -np.inf
    return float(-0.5 * (psi - config.psi0) ** 2 / config.Vpsi)


def sample_psi_csv_ma(
    psi: float,
    U_for_psi: np.ndarray,
    Sig: np.ndarray,
    h: np.ndarray,
    config: BVARConfig,
    rng: np.random.Generator,
    iteration: int,
    cached_precision: Optional[float] = None,
):
    """MH update matching the logic of forecast_BVAR_CSV_t_MA.m.

    MATLAB refreshes the proposal Hessian every 100 iterations with fminunc and
    uses fminbnd in between.  Here the mode is obtained by bounded scalar
    optimization and the local curvature is refreshed at the same cadence.  It
    targets the same conditional posterior while avoiding a dependency on a
    MATLAB-style unconstrained optimizer.
    """

    def lp(x: float) -> float:
        return llike_csv_ma(x, U_for_psi, Sig, h) + _log_prior_psi(x, config)

    res = optimize.minimize_scalar(
        lambda x: -lp(float(x)),
        bounds=(-config.psi_bound, config.psi_bound),
        method="bounded",
        options={"xatol": 1e-8},
    )
    psihat = float(res.x)

    refresh = cached_precision is None or iteration == 1 or iteration % config.psi_hessian_every == 0
    if refresh:
        delta = 1e-3
        lo = max(-config.psi_bound + 1e-6, psihat - delta)
        hi = min(config.psi_bound - 1e-6, psihat + delta)
        if hi <= lo:
            Kpsic = 1.0 / 0.05**2
        else:
            step = min(psihat - lo, hi - psihat)
            if step <= 1e-8:
                Kpsic = 1.0 / 0.05**2
            else:
                f0 = lp(psihat)
                fp = lp(psihat + step)
                fm = lp(psihat - step)
                curvature = -(fm - 2 * f0 + fp) / step**2
                Kpsic = float(curvature) if np.isfinite(curvature) and curvature > 0 else 1.0 / 0.05**2
    else:
        Kpsic = float(cached_precision)

    psic = psihat + rng.standard_normal() / np.sqrt(Kpsic)
    if abs(psic) >= config.psi_bound:
        return float(psi), False, psihat, Kpsic

    log_alpha = (
        lp(psic)
        - lp(psi)
        - 0.5 * (psi - psihat) ** 2 * Kpsic
        + 0.5 * (psic - psihat) ** 2 * Kpsic
    )
    accepted = np.log(rng.random()) < min(0.0, log_alpha)
    return (float(psic) if accepted else float(psi)), bool(accepted), psihat, Kpsic


def _sample_rho(
    h: np.ndarray,
    rho: float,
    sigh2: float,
    config: BVARConfig,
    rng: np.random.Generator,
    bound: float,
):
    Krho = 1.0 / config.Vrho + np.sum(h[:-1] ** 2) / sigh2
    rhohat = (
        config.rho0 / config.Vrho + h[:-1] @ h[1:] / sigh2
    ) / Krho
    rhoc = rhohat + rng.standard_normal() / np.sqrt(Krho)

    def grho(x: float) -> float:
        return float(
            -0.5 * np.log(sigh2 / (1 - x**2))
            - 0.5 * (1 - x**2) / sigh2 * h[0] ** 2
        )

    if abs(rhoc) >= bound:
        return rho, False
    log_alpha = grho(rhoc) - grho(rho)
    accepted = np.log(rng.random()) < min(0.0, log_alpha)
    return (float(rhoc) if accepted else float(rho)), bool(accepted)


def _sample_sigh2(h: np.ndarray, rho: float, config: BVARConfig, rng: np.random.Generator) -> float:
    eh = np.concatenate(
        ([h[0] * np.sqrt(1 - rho**2)], h[1:] - rho * h[:-1])
    )
    shape = config.nuh0 + len(h) / 2
    rate = config.Sh0 + np.sum(eh**2) / 2
    return float(1.0 / rng.gamma(shape, scale=1.0 / rate))


# =============================================================================
# POSTERIOR DRAW GENERATORS
# =============================================================================

def _draw_gaussian_from_precision(
    precision: np.ndarray,
    rhs: np.ndarray,
    rng: np.random.Generator,
):
    C = _safe_cholesky(precision)
    mean = linalg.cho_solve((C, True), rhs)
    z = rng.standard_normal(len(rhs))
    draw = mean + linalg.solve_triangular(C.T, z, lower=False)
    return draw, mean


def _iter_minnesota_draws(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    config: BVARConfig,
    nsims: int,
    burnin: int,
    rng: np.random.Generator,
) -> Iterator[Dict[str, Any]]:
    p = config.p
    Tt, n = shortYt.shape
    k = 1 + n * p
    Z = build_var_design(Y0, shortYt, p)
    ZZ = Z.T @ Z
    prior = prior_minnesota(Y0, shortYt, p, config.c1, config.c2, config.c3)
    Sig = np.diag(prior["sig2"])
    CSig = np.diag(np.sqrt(prior["sig2"]))

    post = []
    for i in range(n):
        sl = slice(i * k, (i + 1) * k)
        V_i = prior["variance"][sl]
        beta0_i = prior["mean"][sl]
        K = np.diag(1.0 / V_i) + ZZ / prior["sig2"][i]
        rhs = beta0_i / V_i + Z.T @ shortYt[:, i] / prior["sig2"][i]
        C = _safe_cholesky(K)
        mean = linalg.cho_solve((C, True), rhs)
        post.append((C, mean))

    for isim in range(1, nsims + burnin + 1):
        A = np.zeros((k, n))
        for i, (C, mean) in enumerate(post):
            A[:, i] = mean + linalg.solve_triangular(
                C.T, rng.standard_normal(k), lower=False
            )
        if isim > burnin:
            yield {
                "A": A,
                "Sig": Sig,
                "CSig": CSig,
                "h_last": 0.0,
                "rho": np.nan,
                "sigh2": np.nan,
                "nu": np.inf,
                "psi": 0.0,
                "et_last": np.zeros(n),
                "gamma_share": np.nan,
                "accept_h": np.nan,
                "accept_rho": np.nan,
                "accept_nu": np.nan,
                "accept_psi": np.nan,
            }


def _draw_joint_coefficients_cg(
    Z: np.ndarray,
    shortYt: np.ndarray,
    Sig: np.ndarray,
    beta0: np.ndarray,
    prior_variance: np.ndarray,
    rng: np.random.Generator,
    rtol: float = 1e-10,
    maxiter: int = 5000,
) -> np.ndarray:
    """Draw beta jointly from its Gaussian conditional without dense SUR matrices.

    Chan's IP/SSVS MATLAB code forms the SUR precision

        P = V^{-1} + Sigma^{-1} \\otimes (Z'Z)

    and uses a sparse Cholesky factor.  SciPy does not ship a sparse Cholesky
    factorization, so we use an equivalent *perturbation-optimization* draw:
    construct a random right-hand side whose covariance is P, then solve the
    same precision system with preconditioned conjugate gradients.  Up to the
    numerical CG tolerance this is a joint draw from exactly the same Gaussian
    conditional; it is not an equation-by-equation approximation.
    """
    Tt, k = Z.shape
    n = shortYt.shape[1]
    ZZ = Z.T @ Z
    ZTY = Z.T @ shortYt
    Omega = np.linalg.inv(Sig)
    prior_prec = 1.0 / prior_variance

    # Posterior mean right-hand side in k x n matrix form.
    rhs_mean = prior_prec * beta0 + ZTY @ Omega

    # Random perturbation with covariance P.
    # prior component: Cov(vec(sqrt(D) z)) = D
    prior_noise = np.sqrt(prior_prec) * rng.standard_normal((k, n))
    # likelihood component: if C'C = Omega then
    # Cov(vec(Z' E C)) = Omega \otimes Z'Z.
    L_omega = _safe_cholesky(Omega)
    C = L_omega.T
    likelihood_noise = Z.T @ rng.standard_normal((Tt, n)) @ C
    rhs = rhs_mean + prior_noise + likelihood_noise

    size = k * n

    def matvec(v: np.ndarray) -> np.ndarray:
        B = np.asarray(v).reshape((k, n), order="F")
        PB = prior_prec * B + ZZ @ B @ Omega
        return PB.reshape(size, order="F")

    operator = sparse_linalg.LinearOperator((size, size), matvec=matvec, dtype=float)

    # Diagonal preconditioner of the full precision.
    diagP = prior_prec + np.diag(ZZ)[:, None] * np.diag(Omega)[None, :]
    inv_diag = 1.0 / diagP.reshape(size, order="F")
    preconditioner = sparse_linalg.LinearOperator(
        (size, size), matvec=lambda v: inv_diag * v, dtype=float
    )

    draw_vec, info = sparse_linalg.cg(
        operator,
        rhs.reshape(size, order="F"),
        M=preconditioner,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
    )
    if info != 0:
        raise RuntimeError(
            f"Joint coefficient CG solve did not converge (info={info})."
        )
    return draw_vec.reshape((k, n), order="F")


def _iter_ip_ssvs_draws(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    config: BVARConfig,
    nsims: int,
    burnin: int,
    rng: np.random.Generator,
) -> Iterator[Dict[str, Any]]:
    """Source-equivalent IP/SSVS Gibbs sampler with joint beta draws."""
    p = config.p
    Tt, n = shortYt.shape
    k = 1 + n * p
    Z = build_var_design(Y0, shortYt, p)

    minn = prior_minnesota(Y0, shortYt, p, config.c1, config.c2, config.c3)
    beta0 = minn["mean"].reshape((k, n), order="F")
    Vbase = minn["variance"].reshape((k, n), order="F")
    S0 = np.diag(minn["sig2"])
    nu0 = n + 3

    Sig = S0.copy()
    gam = np.zeros((k, n), dtype=bool)
    A = beta0.copy()

    for isim in range(1, nsims + burnin + 1):
        if config.prior == "ssvs":
            V = np.where(gam, config.kappa1, Vbase)
        else:
            V = Vbase

        # Joint Gaussian draw: same conditional posterior as the SUR-Cholesky
        # draw in forecast_BVAR_IP.m / forecast_BVAR_SSVS.m.
        A = _draw_joint_coefficients_cg(Z, shortYt, Sig, beta0, V, rng)

        U = shortYt - Z @ A
        Sig = _draw_inverse_wishart(S0 + U.T @ U, nu0 + Tt, rng)

        if config.prior == "ssvs":
            beta = A.reshape(-1, order="F")
            v0 = Vbase.reshape(-1, order="F")
            logp0 = (
                np.log(1 - config.q)
                - 0.5 * (np.log(2 * np.pi * v0) + beta**2 / v0)
            )
            logp1 = (
                np.log(config.q)
                - 0.5
                * (
                    np.log(2 * np.pi * config.kappa1)
                    + beta**2 / config.kappa1
                )
            )
            prob1 = special.expit(logp1 - logp0)
            gam = (rng.random(k * n) < prob1).reshape((k, n), order="F")

        if isim > burnin:
            CSig = _safe_cholesky(Sig)
            yield {
                "A": A.copy(),
                "Sig": Sig.copy(),
                "CSig": CSig,
                "h_last": 0.0,
                "rho": np.nan,
                "sigh2": np.nan,
                "nu": np.inf,
                "psi": 0.0,
                "et_last": np.zeros(n),
                "gamma_share": float(gam.mean()) if config.prior == "ssvs" else np.nan,
                "accept_h": np.nan,
                "accept_rho": np.nan,
                "accept_nu": np.nan,
                "accept_psi": np.nan,
            }

def _iter_ncp_family_draws(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    config: BVARConfig,
    nsims: int,
    burnin: int,
    rng: np.random.Generator,
) -> Iterator[Dict[str, Any]]:
    """Unified NCP/CSV/CSV-t/CSV-t-MA sampler."""
    p = config.p
    Tt, n = shortYt.shape
    k = 1 + n * p
    Z = build_var_design(Y0, shortYt, p)
    prior = prior_natural_conjugate(Y0, shortYt, p, config.c1, config.c2)
    A0, VA0, S0, nu0 = prior["A0"], prior["VA0"], prior["S0"], prior["nu0"]
    iVA0 = 1.0 / VA0

    h = np.zeros(Tt)
    nu = float(config.nu_init)
    rho = float(config.rho_init)
    sigh2 = float(config.sigh2_init)
    lam = np.ones(Tt)
    if config.student_t:
        lam = 1.0 / rng.gamma(nu / 2, scale=2.0 / nu, size=Tt)
    psi = float(config.psi_init if config.ma1 else 0.0)
    psi_precision: Optional[float] = None
    Sig = S0.copy()

    for isim in range(1, nsims + burnin + 1):
        # -------------------------------------------------------------
        # sample Sigma and A
        # -------------------------------------------------------------
        if config.ma1:
            Ztld = _ma_inverse_filter(Z, psi)
            Ytld = _ma_inverse_filter(shortYt, psi)
        else:
            Ztld = Z
            Ytld = shortYt

        weights = np.ones(Tt)
        if config.stochastic_volatility:
            weights *= np.exp(-h)
        if config.student_t:
            weights /= lam
        if config.ma1:
            weights[0] /= (1.0 + psi**2)

        ZiO = Ztld.T * weights
        KA = np.diag(iVA0) + ZiO @ Ztld
        rhs = iVA0[:, None] * A0 + ZiO @ Ytld
        CKA = _safe_cholesky(KA)
        Ahat = linalg.cho_solve((CKA, True), rhs)
        Shat = (
            S0
            + A0.T @ (iVA0[:, None] * A0)
            + Ytld.T @ (weights[:, None] * Ytld)
            - Ahat.T @ KA @ Ahat
        )
        Shat = (Shat + Shat.T) / 2
        Sig = _draw_inverse_wishart(Shat, nu0 + Tt, rng)
        CSig = _safe_cholesky(Sig)
        A = Ahat + linalg.solve_triangular(
            CKA.T, rng.standard_normal((k, n)), lower=False
        ) @ CSig.T

        U = shortYt - Z @ A
        Utld = _ma_inverse_filter(U, psi) if config.ma1 else U
        tmp = _whiten_rows(Utld, CSig)

        accept_h = accept_rho = accept_nu = accept_psi = np.nan

        # -------------------------------------------------------------
        # Chan uses a slightly different Gibbs ordering when MA is on.
        # Preserve that ordering here.
        # -------------------------------------------------------------
        if config.ma1:
            if config.student_t:
                s2_lam = np.sum(tmp**2, axis=1) / np.exp(h)
                lam = 1.0 / rng.gamma(
                    (n + nu) / 2,
                    scale=2.0 / (s2_lam + nu),
                )

            if config.stochastic_volatility:
                s2_h = np.sum(tmp**2, axis=1) / lam
                s2_h[0] /= (1.0 + psi**2)
                h, accept_h = sample_h(s2_h, rho, sigh2, h, n, rng)
                sigh2 = _sample_sigh2(h, rho, config, rng)
                rho, accept_rho = _sample_rho(h, rho, sigh2, config, rng, bound=0.99)

            U_for_psi = U / np.sqrt(lam)[:, None] if config.student_t else U
            psi, accept_psi, _, psi_precision = sample_psi_csv_ma(
                psi, U_for_psi, Sig, h, config, rng, isim, psi_precision
            )

            if config.student_t:
                nu, accept_nu = sample_nu(lam, nu, config.nu_upper, rng)

        else:
            if config.stochastic_volatility:
                s2_h = np.sum(tmp**2, axis=1) / (lam if config.student_t else 1.0)
                h, accept_h = sample_h(s2_h, rho, sigh2, h, n, rng)

            if config.student_t:
                U = shortYt - Z @ A
                tmp = _whiten_rows(U, CSig)
                s2_lam = np.sum(tmp**2, axis=1) / np.exp(h)
                lam = 1.0 / rng.gamma(
                    (n + nu) / 2,
                    scale=2.0 / (s2_lam + nu),
                )
                nu, accept_nu = sample_nu(lam, nu, config.nu_upper, rng)

            if config.stochastic_volatility:
                sigh2 = _sample_sigh2(h, rho, config, rng)
                rho_bound = 0.99 if config.student_t else 0.999
                rho, accept_rho = _sample_rho(h, rho, sigh2, config, rng, bound=rho_bound)

        if isim > burnin:
            U_now = shortYt - Z @ A
            E = _ma_inverse_filter(U_now, psi) if config.ma1 else U_now
            yield {
                "A": A,
                "Sig": Sig,
                "CSig": CSig,
                "h_last": float(h[-1]) if config.stochastic_volatility else 0.0,
                "rho": float(rho) if config.stochastic_volatility else np.nan,
                "sigh2": float(sigh2) if config.stochastic_volatility else np.nan,
                "nu": float(nu) if config.student_t else np.inf,
                "psi": float(psi) if config.ma1 else 0.0,
                "et_last": E[-1].copy() if config.ma1 else np.zeros(n),
                "gamma_share": np.nan,
                "accept_h": accept_h,
                "accept_rho": accept_rho,
                "accept_nu": accept_nu,
                "accept_psi": accept_psi,
            }


def _posterior_iterator(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    config: BVARConfig,
    nsims: int,
    burnin: int,
    rng: np.random.Generator,
) -> Iterator[Dict[str, Any]]:
    validate_config(config)
    if config.prior == "minnesota":
        yield from _iter_minnesota_draws(Y0, shortYt, config, nsims, burnin, rng)
    elif config.prior in {"ip", "ssvs"}:
        yield from _iter_ip_ssvs_draws(Y0, shortYt, config, nsims, burnin, rng)
    elif config.prior == "ncp":
        yield from _iter_ncp_family_draws(Y0, shortYt, config, nsims, burnin, rng)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported prior: {config.prior}")


# =============================================================================
# ONE-SAMPLE ESTIMATION / DIAGNOSTICS
# =============================================================================

def fit_bvar(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    model: str | BVARConfig = "BVAR-CSV-t-MA",
    *,
    nsims: int = 1000,
    burnin: int = 100,
    seed: int = 42,
    store_A: bool = False,
    store_Sigma: bool = False,
    store_structural: bool = False,
    variable_names: Optional[Sequence[str]] = None,
    **overrides: Any,
):
    """Estimate one BVAR sample and retain posterior diagnostics.

    Parameters
    ----------
    store_A : bool
        Store every retained VAR coefficient matrix ``A``.
    store_Sigma : bool
        Store every retained covariance matrix ``Sigma``.
    store_structural : bool
        Convenience switch used by the notebook structural sections.  It turns
        on both ``store_A`` and ``store_Sigma`` and also retains the final MA
        innovation needed to simulate posterior forecast paths without
        re-estimating the model.
    variable_names : sequence[str] or None
        Optional labels attached to the fit.  For the Chan 20-variable sample,
        the standard labels are filled automatically.

    Notes
    -----
    All equations are always fitted.  Storage switches only control which
    posterior draws are retained after estimation; they never reduce the
    dimension of the BVAR.
    """
    config = get_model_config(model, **overrides)
    rng = np.random.default_rng(seed)
    Y0 = np.asarray(Y0, dtype=float)
    shortYt = np.asarray(shortYt, dtype=float)
    n = shortYt.shape[1]
    k = 1 + n * config.p

    if Y0.ndim != 2 or shortYt.ndim != 2:
        raise ValueError("Y0 and shortYt must be two-dimensional arrays.")
    if Y0.shape[1] != n:
        raise ValueError("Y0 and shortYt must contain the same variables.")

    if variable_names is None:
        if n == len(FULL_VARIABLE_LABELS):
            variable_names = FULL_VARIABLE_LABELS.copy()
        elif n == len(CORE_LABELS):
            variable_names = CORE_LABELS.copy()
        else:
            variable_names = [f"var{i + 1}" for i in range(n)]
    else:
        variable_names = list(variable_names)
        if len(variable_names) != n:
            raise ValueError("variable_names must have one label per BVAR variable.")

    if store_structural:
        store_A = True
        store_Sigma = True

    scalars = {
        "rho": np.full(nsims, np.nan),
        "sigh2": np.full(nsims, np.nan),
        "nu": np.full(nsims, np.nan),
        "psi": np.full(nsims, np.nan),
        "h_last": np.full(nsims, np.nan),
        "gamma_share": np.full(nsims, np.nan),
        "accept_h": np.full(nsims, np.nan),
        "accept_rho": np.full(nsims, np.nan),
        "accept_nu": np.full(nsims, np.nan),
        "accept_psi": np.full(nsims, np.nan),
    }
    sigma_diag = np.zeros((nsims, n))
    A_draws = np.zeros((nsims, k, n)) if store_A else None
    Sigma_draws = np.zeros((nsims, n, n)) if store_Sigma else None
    et_last_draws = np.zeros((nsims, n)) if store_structural else None

    active_scalar = {
        "rho": config.stochastic_volatility,
        "sigh2": config.stochastic_volatility,
        "nu": config.student_t,
        "psi": config.ma1,
        "h_last": config.stochastic_volatility,
        "gamma_share": config.prior == "ssvs",
        "accept_h": config.stochastic_volatility,
        "accept_rho": config.stochastic_volatility,
        "accept_nu": config.student_t,
        "accept_psi": config.ma1,
    }

    for s, state in enumerate(
        _posterior_iterator(Y0, shortYt, config, nsims, burnin, rng)
    ):
        for key in scalars:
            if active_scalar[key]:
                value = state[key]
                scalars[key][s] = float(value) if np.ndim(value) == 0 else np.nan
        sigma_diag[s] = np.diag(state["Sig"])
        if store_A:
            A_draws[s] = state["A"]
        if store_Sigma:
            Sigma_draws[s] = state["Sig"]
        if store_structural:
            et_last_draws[s] = state["et_last"]

    summary_rows = []
    for parameter in ("rho", "sigh2", "nu", "psi", "h_last", "gamma_share"):
        values = scalars[parameter]
        finite = values[np.isfinite(values)]
        if len(finite):
            summary_rows.append({
                "parameter": parameter,
                "mean": finite.mean(),
                "sd": finite.std(ddof=1) if len(finite) > 1 else 0.0,
                "p05": np.quantile(finite, 0.05),
                "p50": np.quantile(finite, 0.50),
                "p95": np.quantile(finite, 0.95),
            })

    acceptance = {}
    for parameter, key in (
        ("h", "accept_h"),
        ("rho", "accept_rho"),
        ("nu", "accept_nu"),
        ("psi", "accept_psi"),
    ):
        values = scalars[key]
        finite = values[np.isfinite(values)]
        if len(finite):
            acceptance[parameter] = float(np.mean(finite))

    draws = {
        **scalars,
        "sigma_diag": sigma_diag,
        "A": A_draws,
        "Sigma": Sigma_draws,
        "et_last": et_last_draws,
    }

    fit = {
        "model_name": config.name,
        "config": config,
        "n": n,
        "k": k,
        "p": config.p,
        "nsims": nsims,
        "burnin": burnin,
        "seed": seed,
        "variables": variable_names,
        "Y0": Y0.copy(),
        "shortYt": shortYt.copy(),
        "draws": draws,
        "summary": pd.DataFrame(summary_rows),
        "acceptance": acceptance,
    }

    if store_A:
        fit["coefficient_summary"] = coefficient_posterior_table(
            fit, variable_names=variable_names
        )

    return fit


def coefficient_posterior_table(
    fit: Mapping[str, Any],
    variable_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Posterior summary of every coefficient in every fitted equation."""
    A_draws = fit.get("draws", {}).get("A")
    if A_draws is None:
        raise ValueError("Coefficient draws were not stored. Use store_A=True.")
    A_draws = np.asarray(A_draws, dtype=float)
    if A_draws.ndim != 3:
        raise ValueError("Stored A draws must have shape (draw, k, n).")

    _, k, n = A_draws.shape
    p = int(fit["p"])
    names = list(variable_names or fit.get("variables", []))
    if len(names) != n:
        names = [f"var{i + 1}" for i in range(n)]
    coef_names = coefficient_names(names, p=p)
    if len(coef_names) != k:
        coef_names = ["const"] + [f"coef_{i}" for i in range(1, k)]

    mean = A_draws.mean(axis=0)
    sd = A_draws.std(axis=0, ddof=1)
    p05, p50, p95 = np.quantile(A_draws, [0.05, 0.50, 0.95], axis=0)

    rows = []
    for eq, equation in enumerate(names):
        for r, coefficient in enumerate(coef_names):
            rows.append({
                "equation": equation,
                "coefficient": coefficient,
                "mean": mean[r, eq],
                "sd": sd[r, eq],
                "p05": p05[r, eq],
                "p50": p50[r, eq],
                "p95": p95[r, eq],
            })
    return pd.DataFrame(rows)


# =============================================================================
# FORECASTING
# =============================================================================

def forecast_regressor(shortYt: np.ndarray, p: int) -> np.ndarray:
    """Return [1, y_t, y_{t-1}, ..., y_{t-p+1}]."""
    n = shortYt.shape[1]
    x = np.ones(1 + n * p)
    for lag in range(1, p + 1):
        x[1 + (lag - 1) * n:1 + lag * n] = shortYt[-lag]
    return x


def update_regressor(x: np.ndarray, y_new: np.ndarray, n: int) -> np.ndarray:
    """Equivalent to MATLAB [1 Ytp1 xtp1(2:end-n)]."""
    return np.concatenate(([1.0], np.asarray(y_new), x[1:-n]))


def _gaussian_log_density(
    y: np.ndarray,
    mean: np.ndarray,
    Sig: np.ndarray,
    h: float,
):
    n = len(y)
    dSig = np.exp(h) * np.diag(Sig)
    lden = -0.5 * np.log(2 * np.pi * dSig) - 0.5 * (y - mean) ** 2 / dSig
    CSig = _safe_cholesky(Sig)
    tmpu = linalg.solve_triangular(CSig, (y - mean), lower=True)
    joint = (
        -n / 2 * np.log(2 * np.pi)
        - n / 2 * h
        - np.sum(np.log(np.diag(CSig)))
        - 0.5 * (tmpu @ tmpu) / np.exp(h)
    )
    return lden, float(joint)


def _student_t_log_density(
    y: np.ndarray,
    mean: np.ndarray,
    Sig: np.ndarray,
    h: float,
    nu: float,
):
    n = len(y)
    dSig = np.exp(h) * np.diag(Sig)
    ct = (
        special.gammaln((nu + 1) / 2)
        - special.gammaln(nu / 2)
        - 0.5 * np.log(nu * np.pi * dSig)
    )
    lden = ct - (nu + 1) / 2 * np.log(
        1 + (y - mean) ** 2 / dSig / nu
    )

    CSig = _safe_cholesky(Sig)
    tmpu = linalg.solve_triangular(CSig, (y - mean), lower=True)
    ct_joint = (
        special.gammaln((nu + n) / 2)
        - special.gammaln(nu / 2)
        - n / 2 * np.log(nu * np.pi)
    )
    joint = (
        ct_joint
        - np.sum(np.log(np.diag(CSig)))
        - n / 2 * h
        - (nu + n) / 2 * np.log(1 + (tmpu @ tmpu) / nu / np.exp(h))
    )
    return lden, float(joint)


def _draw_future_innovation(
    Sig: np.ndarray,
    CSig: np.ndarray,
    h: float,
    nu: float,
    student_t: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    n = Sig.shape[0]
    shock = np.exp(h / 2) * (CSig @ rng.standard_normal(n))
    if student_t:
        shock = shock / np.sqrt(rng.gamma(nu / 2, scale=2.0 / nu))
    return shock


def _forecast_from_state(
    state: Mapping[str, Any],
    shortYt: np.ndarray,
    data_tpk: np.ndarray,
    is_last_miss: bool,
    t: int,
    T: int,
    config: BVARConfig,
    rng: np.random.Generator,
):
    A = state["A"]
    Sig = state["Sig"]
    CSig = state["CSig"]
    n = Sig.shape[0]
    x = forecast_regressor(shortYt, config.p)

    psi = float(state["psi"]) if config.ma1 else 0.0
    etp1 = np.asarray(state["et_last"], dtype=float).copy() if config.ma1 else np.zeros(n)
    htp1 = float(state["h_last"]) if config.stochastic_volatility else 0.0
    rho = float(state["rho"]) if config.stochastic_volatility else 0.0
    sigh2 = float(state["sigh2"]) if config.stochastic_volatility else 0.0
    nu = float(state["nu"]) if config.student_t else np.inf

    def evolve_h(current: float) -> float:
        if not config.stochastic_volatility:
            return 0.0
        return rho * current + np.sqrt(sigh2) * rng.standard_normal()

    if is_last_miss:
        htp1 = evolve_h(htp1)
        mean = x @ A + psi * etp1
        innovation = _draw_future_innovation(
            Sig, CSig, htp1, nu, config.student_t, rng
        )
        if config.ma1:
            etp1 = innovation
        y_new = mean + innovation
        x = update_regressor(x, y_new, n)

    out = []
    for tt in (1, 2):
        htp1 = evolve_h(htp1)
        mean = x @ A + psi * etp1
        if config.student_t:
            lden, joint = _student_t_log_density(
                data_tpk[tt - 1], mean, Sig, htp1, nu
            )
        else:
            lden, joint = _gaussian_log_density(
                data_tpk[tt - 1], mean, Sig, htp1
            )

        # Preserve the exact guard in the supplied MATLAB forecast scripts.
        valid = tt == 1 or (tt == 2 and t <= T - tt)
        out.append((mean.copy(), lden.copy(), joint, valid))

        innovation = _draw_future_innovation(
            Sig, CSig, htp1, nu, config.student_t, rng
        )
        if config.ma1:
            etp1 = innovation
        y_new = mean + innovation
        x = update_regressor(x, y_new, n)

    return out


def forecast_bvar_origin(
    Y0: np.ndarray,
    shortYt: np.ndarray,
    data_tpk: np.ndarray,
    is_last_miss: bool,
    t: int,
    T: int,
    model: str | BVARConfig = "BVAR-CSV-t-MA",
    *,
    nsims: int = 5000,
    burnin: int = 100,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    **overrides: Any,
):
    """Forecast one recursive origin for any of the seven large BVARs."""
    config = get_model_config(model, **overrides)
    if rng is None:
        rng = np.random.default_rng(seed)

    n = shortYt.shape[1]
    tmp0 = np.zeros((nsims, 2 * n + 1))
    tmp1 = np.zeros((nsims, 2 * n + 1))

    for s, state in enumerate(
        _posterior_iterator(Y0, shortYt, config, nsims, burnin, rng)
    ):
        forecasts = _forecast_from_state(
            state, shortYt, data_tpk, is_last_miss, t, T, config, rng
        )
        mean0, lden0, joint0, _ = forecasts[0]
        tmp0[s] = np.concatenate([mean0, lden0, [joint0]])

        mean1, lden1, joint1, valid1 = forecasts[1]
        if valid1:
            tmp1[s] = np.concatenate([mean1, lden1, [joint1]])

    return tmp0, tmp1


def aggregate_forecast_draws(tmp: np.ndarray, n: int):
    """Posterior mean forecast and log predictive likelihood via log-mean-exp."""
    tmp = np.asarray(tmp, dtype=float)
    point = tmp[:, :n].mean(axis=0)
    logdens = tmp[:, n:]  # n marginal densities + one joint density
    max_logdens = np.max(logdens, axis=0)
    lpl = np.log(np.mean(np.exp(logdens - max_logdens), axis=0)) + max_logdens
    return point, lpl


# =============================================================================
# FULL RECURSIVE EXERCISE
# =============================================================================

def run_large_bvar(
    rt_data,
    nonrev_data,
    tcode,
    var_type,
    model: str | BVARConfig = "BVAR-CSV-t-MA",
    *,
    p: Optional[int] = None,
    T0: int = 41,
    T: int = 208,
    nsims: int = 5000,
    burnin: int = 100,
    seed: int = 0,
    progress: bool = True,
    **overrides: Any,
):
    """Replicate Chan's recursive nowcast / 1-quarter-ahead exercise."""
    config = get_model_config(model, **overrides)
    if p is not None:
        config = replace(config, p=p)
        validate_config(config)

    rng = np.random.default_rng(seed)
    n = len(var_type)
    yhat0 = np.zeros((T - T0 + 1, 3 * n + 1))
    yhat1 = np.zeros((T - 1 - T0 + 1, 3 * n + 1))

    for t in range(T0, T + 1):
        if progress:
            print(f"{T - t + 1} recursive origins remaining — {config.name}", end="\r")

        data_t, data_tpk = loaddata(rt_data, nonrev_data, t, T0, tcode, var_type)
        data_t, is_last_miss = trim_data(data_t)
        Y0 = data_t[:config.p]
        shortYt = data_t[config.p:]

        tmp0, tmp1 = forecast_bvar_origin(
            Y0,
            shortYt,
            data_tpk,
            is_last_miss,
            t,
            T,
            config,
            nsims=nsims,
            burnin=burnin,
            rng=rng,
        )

        point0, lpl0 = aggregate_forecast_draws(tmp0, n)
        yhat0[t - T0] = np.concatenate([data_tpk[0], point0, lpl0])

        if t <= T - 1:
            point1, lpl1 = aggregate_forecast_draws(tmp1, n)
            yhat1[t - T0] = np.concatenate([data_tpk[1], point1, lpl1])

    if progress:
        print(" " * 100, end="\r")

    core = np.asarray(CORE_VARIABLES, dtype=int)
    obs0 = yhat0[4:, core]
    fc0 = yhat0[4:, n + core]
    lpl0 = yhat0[4:, 2 * n + core]

    obs1 = yhat1[3:, core]
    fc1 = yhat1[3:, n + core]
    lpl1 = yhat1[3:, 2 * n + core]

    rmsfe = np.column_stack([
        np.sqrt(np.mean((obs0 - fc0) ** 2, axis=0)),
        np.sqrt(np.mean((obs1 - fc1) ** 2, axis=0)),
    ])
    alpl = np.column_stack([lpl0.mean(axis=0), lpl1.mean(axis=0)])

    return {
        "model_name": config.name,
        "config": config,
        "variables": CORE_LABELS.copy(),
        "evaluation_indices": CORE_VARIABLES.copy(),
        "full_variables": FULL_VARIABLE_LABELS.copy(),
        "rmsfe": rmsfe,
        "alpl": alpl,
        "yhat0": yhat0,
        "yhat1": yhat1,
        "n": n,
        "p": config.p,
        "k": 1 + n * config.p,
        "T0": T0,
        "T": T,
        "nsims": nsims,
        "burnin": burnin,
        "seed": seed,
    }


def result_tables(result: Mapping[str, Any]):
    """RMSFE and ALPL tables for GDP, IP, unemployment and PCE inflation."""
    columns = ["Nowcast", "1-quarter-ahead"]
    rmsfe = pd.DataFrame(result["rmsfe"], index=result["variables"], columns=columns)
    alpl = pd.DataFrame(result["alpl"], index=result["variables"], columns=columns)
    return rmsfe, alpl


def compare_result_tables(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Long-form comparison table for several completed recursive runs."""
    rows = []
    for result in results:
        for i, variable in enumerate(result["variables"]):
            rows.extend([
                {
                    "model": result["model_name"],
                    "variable": variable,
                    "horizon": "Nowcast",
                    "RMSFE": result["rmsfe"][i, 0],
                    "ALPL": result["alpl"][i, 0],
                },
                {
                    "model": result["model_name"],
                    "variable": variable,
                    "horizon": "1-quarter-ahead",
                    "RMSFE": result["rmsfe"][i, 1],
                    "ALPL": result["alpl"][i, 1],
                },
            ])
    return pd.DataFrame(rows)


# =============================================================================
# LEGACY BVAR-SMALL API (kept so the previous notebook/imports do not break)
# =============================================================================


# =============================================================================
# POSTERIOR FORECAST FANS AND STRUCTURAL ANALYSIS
# =============================================================================

def _require_structural_draws(fit: Mapping[str, Any]):
    """Validate and return stored A/Sigma draws used by structural tools."""
    draws = fit.get("draws", {})
    A_draws = draws.get("A")
    Sigma_draws = draws.get("Sigma")
    if A_draws is None or Sigma_draws is None:
        raise ValueError(
            "Structural analysis requires stored A and Sigma draws. "
            "Estimate with fit_bvar(..., store_structural=True)."
        )
    A_draws = np.asarray(A_draws, dtype=float)
    Sigma_draws = np.asarray(Sigma_draws, dtype=float)
    if A_draws.ndim != 3:
        raise ValueError("A draws must have shape (draw, k, n).")
    if Sigma_draws.ndim != 3:
        raise ValueError("Sigma draws must have shape (draw, n, n).")
    if len(A_draws) != len(Sigma_draws):
        raise ValueError("A and Sigma must contain the same number of draws.")
    return A_draws, Sigma_draws


def _draw_psi(fit: Mapping[str, Any], draw_index: int) -> float:
    """Return the MA coefficient for one draw, or zero when MA is inactive."""
    config = fit["config"]
    if not config.ma1:
        return 0.0
    values = np.asarray(fit["draws"]["psi"], dtype=float)
    value = float(values[draw_index])
    if not np.isfinite(value):
        raise ValueError("Stored psi draw is not finite for an MA(1) model.")
    return value


def _structural_lag_matrices(A: np.ndarray, n: int, p: int) -> list[np.ndarray]:
    """Return A_1,...,A_p in response-by-explanatory orientation."""
    return [
        A[1 + lag * n:1 + (lag + 1) * n, :].T
        for lag in range(p)
    ]


def _structural_is_stable(A: np.ndarray, n: int, p: int) -> bool:
    """Strict VAR stability check for one coefficient draw."""
    top = np.hstack(_structural_lag_matrices(A, n, p))
    if p == 1:
        F = top
    else:
        bottom = np.hstack([
            np.eye(n * (p - 1)),
            np.zeros((n * (p - 1), n)),
        ])
        F = np.vstack([top, bottom])
    return bool(np.max(np.abs(np.linalg.eigvals(F))) < 1.0)


def _structural_irf_tensor(
    A: np.ndarray,
    impact_matrix: np.ndarray,
    p: int,
    H: int,
    psi: float = 0.0,
) -> np.ndarray:
    """Structural responses for a VAR(p) with Chan's optional scalar MA(1).

    ``impact_matrix`` is in column convention: column j is the impact of
    structural innovation j and ``impact_matrix @ impact_matrix.T = Sigma``.
    For the MA model, the same innovation also enters the next reduced-form
    residual with coefficient ``psi``.
    """
    if H < 1:
        raise ValueError("H must be at least 1.")
    n = impact_matrix.shape[0]
    lags = _structural_lag_matrices(A, n, p)
    theta = np.zeros((H, n, n), dtype=float)
    theta[0] = impact_matrix
    for h in range(1, H):
        for lag in range(1, min(p, h) + 1):
            theta[h] += lags[lag - 1] @ theta[h - lag]
        if h == 1 and psi != 0.0:
            theta[h] += psi * impact_matrix
    return theta



def forecast_origin_from_fit(
    fit: Mapping[str, Any],
    shortYt: np.ndarray,
    data_tpk: np.ndarray,
    is_last_miss: bool,
    *,
    t: int,
    T: int,
    seed: int = 123,
):
    """Chan-style nowcast / one-quarter predictive output from stored draws.

    Unlike :func:`forecast_bvar_origin`, this function does not rerun the
    sampler.  It evaluates the predictive distribution using the exact draws
    already retained by ``fit_bvar(..., store_structural=True)``.
    """
    A_draws, Sigma_draws = _require_structural_draws(fit)
    config = fit["config"]
    shortYt = np.asarray(shortYt, dtype=float)
    data_tpk = np.asarray(data_tpk, dtype=float)
    n = shortYt.shape[1]
    nsims = len(A_draws)
    tmp0 = np.zeros((nsims, 2 * n + 1), dtype=float)
    tmp1 = np.zeros((nsims, 2 * n + 1), dtype=float)
    et_store = fit["draws"].get("et_last")
    rng = np.random.default_rng(seed)

    for d, (A, Sig) in enumerate(zip(A_draws, Sigma_draws)):
        state = {
            "A": A,
            "Sig": Sig,
            "CSig": _safe_cholesky(0.5 * (Sig + Sig.T)),
            "h_last": (
                float(fit["draws"]["h_last"][d])
                if config.stochastic_volatility else 0.0
            ),
            "rho": (
                float(fit["draws"]["rho"][d])
                if config.stochastic_volatility else 0.0
            ),
            "sigh2": (
                float(fit["draws"]["sigh2"][d])
                if config.stochastic_volatility else 0.0
            ),
            "nu": (
                float(fit["draws"]["nu"][d])
                if config.student_t else np.inf
            ),
            "psi": _draw_psi(fit, d),
            "et_last": (
                np.asarray(et_store[d], dtype=float)
                if config.ma1 else np.zeros(n)
            ),
        }
        forecasts = _forecast_from_state(
            state, shortYt, data_tpk, is_last_miss, t, T, config, rng
        )
        mean0, lden0, joint0, _ = forecasts[0]
        tmp0[d] = np.concatenate([mean0, lden0, [joint0]])
        mean1, lden1, joint1, valid1 = forecasts[1]
        if valid1:
            tmp1[d] = np.concatenate([mean1, lden1, [joint1]])
    return tmp0, tmp1

def forecast_paths_from_fit(
    fit: Mapping[str, Any],
    shortYt: Optional[np.ndarray] = None,
    *,
    H: int = 12,
    seed: int = 123,
    is_last_miss: bool = False,
) -> Dict[str, Any]:
    """Simulate posterior predictive paths from an already estimated fit.

    This reuses the retained parameter draws, so the fan chart does not rerun
    the Gibbs/MH sampler.  CSV volatility, Student-t tails and the MA(1)
    innovation state are propagated with the same state equations used by the
    Chan forecast routines.
    """
    if H < 1:
        raise ValueError("H must be at least 1.")
    A_draws, Sigma_draws = _require_structural_draws(fit)
    config = fit["config"]
    Y = np.asarray(shortYt if shortYt is not None else fit["shortYt"], dtype=float)
    n = Y.shape[1]
    if A_draws.shape[2] != n:
        raise ValueError("shortYt variable dimension does not match the fitted model.")

    rng = np.random.default_rng(seed)
    paths = np.zeros((len(A_draws), H, n), dtype=float)
    et_store = fit["draws"].get("et_last")

    for d, (A, Sig) in enumerate(zip(A_draws, Sigma_draws)):
        CSig = _safe_cholesky(0.5 * (Sig + Sig.T))
        x = forecast_regressor(Y, config.p)
        psi = _draw_psi(fit, d)
        if config.ma1:
            if et_store is None:
                raise ValueError(
                    "MA forecast paths require et_last. Use store_structural=True."
                )
            et_last = np.asarray(et_store[d], dtype=float).copy()
        else:
            et_last = np.zeros(n)

        if config.stochastic_volatility:
            h = float(fit["draws"]["h_last"][d])
            rho = float(fit["draws"]["rho"][d])
            sigh2 = float(fit["draws"]["sigh2"][d])
        else:
            h = rho = sigh2 = 0.0

        nu = (
            float(fit["draws"]["nu"][d])
            if config.student_t
            else np.inf
        )

        def evolve_h(current: float) -> float:
            if not config.stochastic_volatility:
                return 0.0
            return rho * current + np.sqrt(sigh2) * rng.standard_normal()

        # Match Chan's treatment of an incomplete latest real-time row: one
        # latent transition is simulated before the first reported horizon.
        if is_last_miss:
            h = evolve_h(h)
            mean = x @ A + psi * et_last
            innovation = _draw_future_innovation(
                Sig, CSig, h, nu, config.student_t, rng
            )
            y_new = mean + innovation
            if config.ma1:
                et_last = innovation
            x = update_regressor(x, y_new, n)

        for hh in range(H):
            h = evolve_h(h)
            mean = x @ A + psi * et_last
            innovation = _draw_future_innovation(
                Sig, CSig, h, nu, config.student_t, rng
            )
            y_new = mean + innovation
            paths[d, hh] = y_new
            if config.ma1:
                et_last = innovation
            x = update_regressor(x, y_new, n)

    return {
        "model_name": fit["model_name"],
        "paths": paths,
        "history": Y,
        "variables": list(fit.get("variables", [f"var{i+1}" for i in range(n)])),
        "H": H,
        "seed": seed,
        "is_last_miss": bool(is_last_miss),
    }



# =============================================================================
# MCMC DIAGNOSTICS
# =============================================================================

def ess(x: np.ndarray) -> float:
    """Effective sample size using Geyer's initial-positive-sequence rule.

    This is the same compact diagnostic used in the user's other Bayesian
    modules.  Constant chains have zero Monte Carlo variance, so their ESS is
    reported as the retained draw count rather than producing a 0/0 ACF.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float(n)

    xc = x - x.mean()
    var0 = float(xc @ xc)
    if not np.isfinite(var0) or var0 <= 0.0:
        return float(n)

    f = np.fft.rfft(xc, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n]
    if acov[0] <= 0:
        return float(n)
    rho = acov / acov[0]

    tau = -1.0
    for k_pair in range(n // 2):
        j = 2 * k_pair
        if j + 1 >= n:
            break
        pair = rho[j] + rho[j + 1]
        if not np.isfinite(pair) or pair < 0:
            break
        tau += 2.0 * pair

    return float(n / max(tau, 1.0))


def chain_acf(x: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Autocorrelation of a retained chain, lags 0,...,nlags."""
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.full(int(nlags) + 1, np.nan)
    nlags = min(int(nlags), len(x) - 1)
    xc = x - x.mean()
    var0 = float(xc @ xc)
    if var0 <= 0:
        out = np.zeros(nlags + 1)
        out[0] = 1.0
        return out
    f = np.fft.rfft(xc, 2 * len(x))
    acov = np.fft.irfft(f * np.conj(f))[:nlags + 1]
    return acov / acov[0]


def mcmc_diagnostics(quantities: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """ESS / MCSE / MCSE-to-posterior-sd table for named retained chains."""
    rows = {}
    for name, values in quantities.items():
        d = np.asarray(values, dtype=float).reshape(-1)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue

        posterior_sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        e = ess(d)
        mcse = posterior_sd / np.sqrt(e) if e > 0 else np.nan
        acf1 = chain_acf(d, nlags=1)
        rows[name] = {
            "post. mean": float(d.mean()),
            "post. sd": posterior_sd,
            "ESS": e,
            "ESS/share": e / len(d),
            "MCSE": mcse,
            "MCSE/sd": (mcse / posterior_sd) if posterior_sd > 0 else 0.0,
            "ACF(1)": float(acf1[1]) if len(acf1) > 1 else np.nan,
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def _default_mcmc_probe_indices(fit: Mapping[str, Any]) -> list[int]:
    """Small representative coefficient/covariance set for large-BVAR diagnosis."""
    n = int(fit["n"])
    if n > max(CORE_VARIABLES):
        return list(CORE_VARIABLES)
    return list(range(min(4, n)))


def bvar_mcmc_diagnostic_series(
    fit: Mapping[str, Any],
    probe_indices: Optional[Sequence[int]] = None,
) -> Dict[str, np.ndarray]:
    """Collect the chains that are informative for Gibbs/MH diagnosis.

    A 20-variable VAR(4) has 1,620 VAR coefficients, so plotting every chain
    would obscure rather than diagnose mixing.  The catalogue therefore uses:

    * every active scalar latent/MH parameter;
    * own first-lag coefficients for four representative variables;
    * the matching diagonal covariance elements;
    * log det(Sigma), when full covariance draws were retained.

    The complete coefficient posterior remains available separately through
    ``fit['coefficient_summary']``.
    """
    names = list(fit.get("variables", []))
    n = int(fit["n"])
    if len(names) != n:
        names = [f"var{i + 1}" for i in range(n)]

    if probe_indices is None:
        probe_indices = _default_mcmc_probe_indices(fit)
    probe_indices = [int(i) for i in probe_indices]
    if any(i < 0 or i >= n for i in probe_indices):
        raise ValueError("probe_indices contains a variable outside the BVAR.")

    series: Dict[str, np.ndarray] = {}

    # Active scalar / latent chains.
    for key in ("rho", "sigh2", "nu", "psi", "h_last", "gamma_share"):
        values = np.asarray(fit["draws"].get(key, []), dtype=float)
        if values.size and np.isfinite(values).any():
            series[key] = values

    # Representative coefficient chains.
    A_draws = fit["draws"].get("A")
    if A_draws is not None:
        A_draws = np.asarray(A_draws, dtype=float)
        if A_draws.ndim != 3:
            raise ValueError("fit['draws']['A'] must have shape (draw, k, n).")
        for j in probe_indices:
            # row 0 is the intercept; rows 1:1+n are lag 1.
            series[f"A[{names[j]} <- L1 {names[j]}]"] = A_draws[:, 1 + j, j]

    # Representative variance chains are retained even when full Sigma is not.
    sigma_diag = np.asarray(fit["draws"].get("sigma_diag"), dtype=float)
    if sigma_diag.ndim == 2:
        for j in probe_indices:
            series[f"Sigma[{names[j]},{names[j]}]"] = sigma_diag[:, j]

    Sigma_draws = fit["draws"].get("Sigma")
    if Sigma_draws is not None:
        Sigma_draws = np.asarray(Sigma_draws, dtype=float)
        if Sigma_draws.ndim == 3:
            logdet = np.empty(len(Sigma_draws))
            for d, Sigma in enumerate(Sigma_draws):
                sign, value = np.linalg.slogdet(0.5 * (Sigma + Sigma.T))
                logdet[d] = value if sign > 0 else np.nan
            if np.isfinite(logdet).any():
                series["log det(Sigma)"] = logdet

    return series


def bvar_mcmc_diagnostics(
    fit: Mapping[str, Any],
    probe_indices: Optional[Sequence[int]] = None,
    *,
    acf_lags: int = 40,
) -> Dict[str, Any]:
    """Central MCMC diagnosis for Chan specifications that use Gibbs/MH.

    Minnesota and natural-conjugate BVAR draws are direct/iid posterior draws,
    not Markov-chain sweeps, so this routine deliberately marks them as
    ``is_mcmc=False``.  The Gibbs/MH families are IP, SSVS, CSV, CSV-t and
    CSV-t-MA.
    """
    config = fit["config"]
    is_mcmc = bool(
        config.prior in {"ip", "ssvs"}
        or config.stochastic_volatility
        or config.student_t
        or config.ma1
    )

    if not is_mcmc:
        return {
            "model_name": fit["model_name"],
            "is_mcmc": False,
            "series": {},
            "table": pd.DataFrame(),
            "acceptance": pd.DataFrame(),
            "acf": {},
            "acf_lags": int(acf_lags),
            "note": "Direct/iid posterior sampling; no Markov-chain convergence diagnosis required.",
        }

    series = bvar_mcmc_diagnostic_series(fit, probe_indices=probe_indices)
    if not series:
        raise ValueError(
            "No retained diagnostic chains are available. Estimate with "
            "store_structural=True (or at least store_A=True) for full diagnosis."
        )

    table = mcmc_diagnostics(series)
    acf = {
        name: chain_acf(values, nlags=acf_lags)
        for name, values in series.items()
    }

    acceptance_rows = []
    for block, share in fit.get("acceptance", {}).items():
        acceptance_rows.append(
            {"block": block, "acceptance share": float(share)}
        )
    acceptance = (
        pd.DataFrame(acceptance_rows).set_index("block")
        if acceptance_rows
        else pd.DataFrame(columns=["acceptance share"])
    )

    return {
        "model_name": fit["model_name"],
        "is_mcmc": True,
        "series": series,
        "table": table,
        "acceptance": acceptance,
        "acf": acf,
        "acf_lags": int(acf_lags),
        "n_saved": int(fit["nsims"]),
        "probe_indices": list(probe_indices or _default_mcmc_probe_indices(fit)),
    }


def sign_identification_diagnostics(
    fit: Mapping[str, Any],
    restrictions: Mapping[str, int],
    names: Sequence[str],
    *,
    shock_var: str,
    horizon: int = 1,
    p: Optional[int] = None,
    stable_only: bool = True,
    n_rotations: int = 30,
    max_draws: int = 120,
    max_tries: int = 2000,
    space: str = "whitened",
    seed: int = 42,
    baseline: bool = True,
) -> Dict[str, Any]:
    """Geometric strength diagnostic for one sign-restricted shock.

    This is adapted from the user's existing BVAR sign-identification
    diagnostic rather than reconstructed from scratch.

    The structural draw routine may search all columns of a QR rotation as a
    computational shortcut.  This diagnostic intentionally samples ONE fixed
    Haar direction per proposal (the column indexed by ``shock_var``).  That
    makes the reported acceptance rate a genuine measure of the volume of the
    admissible direction set, independent of the search algorithm used to find
    an accepted A0.

    Per retained posterior draw it records:

    * acceptance rate: volume of the admissible set;
    * mean resultant length R_d: concentration of admissible directions;
    * 90% angular aperture: an interpretable dispersion measure.

    Because the diagnostic conditions on each model's posterior Sigma (and on
    A, plus psi when restrictions extend beyond impact), it is model-specific.
    """
    A_draws, Sigma_draws = _require_structural_draws(fit)
    names = list(names)
    n = int(fit["n"])
    p = int(fit["p"] if p is None else p)

    if len(names) != n:
        raise ValueError("names must contain one label per BVAR variable.")
    if shock_var not in names:
        raise ValueError(f"{shock_var!r} not found in names.")
    if not restrictions:
        raise ValueError("restrictions must contain at least one sign.")
    unknown = [v for v in restrictions if v not in names]
    if unknown:
        raise ValueError(f"Unknown variable(s) in restrictions: {unknown}")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if n_rotations < 4:
        raise ValueError("n_rotations must be at least 4.")
    if max_draws < 1 or max_tries < 1:
        raise ValueError("max_draws and max_tries must be positive.")
    if space not in {"whitened", "impact"}:
        raise ValueError("space must be 'whitened' or 'impact'.")

    signs = np.array(
        [np.sign(restrictions.get(v, 0)) for v in names],
        dtype=float,
    )
    restricted = signs != 0
    if not restricted.any():
        raise ValueError("At least one restriction must be +1 or -1.")

    shock_slot = names.index(shock_var)
    rng = np.random.default_rng(seed)

    n_draws = len(A_draws)
    if max_draws >= n_draws:
        candidate_indices = np.arange(n_draws, dtype=int)
    else:
        candidate_indices = np.unique(
            np.linspace(0, n_draws - 1, int(max_draws), dtype=int)
        )

    def _direction_ok(A: np.ndarray, impact: np.ndarray, psi: float) -> bool:
        responses = np.zeros((horizon, n), dtype=float)
        responses[0] = impact
        lag_matrices = _structural_lag_matrices(A, n, p)
        for h in range(1, horizon):
            for lag in range(1, min(p, h) + 1):
                responses[h] += lag_matrices[lag - 1] @ responses[h - lag]
            if h == 1 and psi != 0.0:
                responses[h] += psi * impact
        return bool(
            np.all(
                np.sign(responses[:, restricted])
                == signs[restricted][None, :]
            )
        )

    R_out = []
    kappa_out = []
    aperture_out = []
    acceptance_out = []
    draw_indices = []
    angles_pool = []
    proposal_counts = []
    dropped_unstable = 0
    dropped_cholesky = 0
    dropped_too_few = 0

    for d in candidate_indices:
        A = A_draws[d]
        if stable_only and not _structural_is_stable(A, n, p):
            dropped_unstable += 1
            continue

        Sigma = 0.5 * (Sigma_draws[d] + Sigma_draws[d].T)
        try:
            P = _safe_cholesky(Sigma)
        except np.linalg.LinAlgError:
            dropped_cholesky += 1
            continue

        psi = _draw_psi(fit, d)
        directions = []
        proposals = 0

        while len(directions) < n_rotations and proposals < max_tries:
            proposals += 1
            K = rng.standard_normal((n, n))
            Q, Rq = np.linalg.qr(K)
            dsign = np.sign(np.diag(Rq))
            dsign[dsign == 0] = 1.0
            Q = Q @ np.diag(dsign)

            impact = P @ Q[:, shock_slot]
            oriented = None
            if _direction_ok(A, impact, psi):
                oriented = impact
            elif _direction_ok(A, -impact, psi):
                oriented = -impact

            if oriented is None:
                continue

            if space == "whitened":
                q = np.linalg.solve(P, oriented)
            else:
                q = oriented
            norm = np.linalg.norm(q)
            if norm <= 0 or not np.isfinite(norm):
                continue
            directions.append(q / norm)

        minimum_required = max(4, n_rotations // 3)
        if len(directions) < minimum_required:
            dropped_too_few += 1
            continue

        q = np.asarray(directions)
        mean_direction = q.mean(axis=0)
        R_d = float(np.linalg.norm(mean_direction))
        if R_d <= 0:
            continue
        qbar = mean_direction / R_d
        angles = np.degrees(
            np.arccos(np.clip(q @ qbar, -1.0, 1.0))
        )

        R_out.append(R_d)
        kappa_out.append(
            R_d * (n - R_d**2) / max(1.0 - R_d**2, 1e-12)
        )
        aperture_out.append(float(np.percentile(angles, 90)))
        acceptance_out.append(len(directions) / proposals)
        draw_indices.append(int(d))
        angles_pool.append(angles)
        proposal_counts.append(proposals)

    if not R_out:
        raise ValueError(
            "No posterior draw yielded enough admissible sign-restricted "
            "directions for identification diagnosis."
        )

    diag = {
        "model_name": fit["model_name"],
        "R": np.asarray(R_out),
        "kappa": np.asarray(kappa_out),
        "aperture90": np.asarray(aperture_out),
        "acceptance": np.asarray(acceptance_out),
        "angles_deg": np.concatenate(angles_pool),
        "draw_indices": np.asarray(draw_indices, dtype=int),
        "proposals": np.asarray(proposal_counts, dtype=int),
        "n_rotations": int(n_rotations),
        "space": space,
        "restrictions": dict(restrictions),
        "horizon": int(horizon),
        "shock_var": shock_var,
        "n_candidates": len(candidate_indices),
        "n_measured": len(R_out),
        "dropped_unstable": dropped_unstable,
        "dropped_cholesky": dropped_cholesky,
        "dropped_too_few": dropped_too_few,
        "baseline": None,
    }

    if baseline:
        baseline_sign = int(np.sign(restrictions.get(shock_var, +1)))
        if baseline_sign == 0:
            baseline_sign = +1
        diag["baseline"] = sign_identification_diagnostics(
            fit,
            {shock_var: baseline_sign},
            names,
            shock_var=shock_var,
            horizon=1,
            p=p,
            stable_only=stable_only,
            n_rotations=n_rotations,
            max_draws=max_draws,
            max_tries=max_tries,
            space=space,
            seed=seed + 1,
            baseline=False,
        )

    return diag


def sign_restriction_draws(
    fit: Mapping[str, Any],
    restrictions: Mapping[str, int],
    names: Sequence[str],
    *,
    p: Optional[int] = None,
    shock_var: Optional[str] = None,
    shock_name: Optional[str] = None,
    horizon: int = 1,
    stable_only: bool = True,
    max_tries: int = 1000,
    allow_flip: bool = True,
    search_all_columns: bool = True,
    max_draws: Optional[int] = 250,
    seed: int = 42,
):
    """Identify one structural shock by QR sign restrictions, draw by draw.

    The candidate impact matrix is ``P @ Q`` with ``P P' = Sigma`` and Q
    orthonormal.  This preserves the reduced-form covariance.  In the MA(1)
    model, restrictions beyond impact use the VAR+MA impulse response recursion.
    """
    A_draws, Sigma_draws = _require_structural_draws(fit)
    names = list(names)
    n = Sigma_draws.shape[1]
    p = int(fit["p"] if p is None else p)
    if len(names) != n:
        raise ValueError("names must contain one label per BVAR variable.")
    if not restrictions:
        raise ValueError("restrictions must contain at least one sign.")
    unknown = [v for v in restrictions if v not in names]
    if unknown:
        raise ValueError(f"Unknown variable(s) in restrictions: {unknown}")
    signs = np.array([np.sign(restrictions.get(v, 0)) for v in names], dtype=float)
    restricted = signs != 0
    if not restricted.any():
        raise ValueError("At least one restriction must be +1 or -1.")

    if shock_var is None:
        positives = [v for v, s in restrictions.items() if s > 0]
        if len(positives) != 1:
            raise ValueError("Pass shock_var explicitly when it cannot be inferred.")
        shock_var = positives[0]
    if shock_var not in names:
        raise ValueError(f"{shock_var!r} not found in names.")
    shock_slot = names.index(shock_var)
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    rng = np.random.default_rng(seed)
    accepted_matrices = []
    used_indices = []
    tries = []
    flipped = 0
    dropped_unstable = 0
    dropped_cholesky = 0
    dropped_max_tries = 0

    def signs_hold(A, impact, psi):
        # Propagate a single impact vector with the exact VAR+MA recursion.
        responses = np.zeros((horizon, n), dtype=float)
        responses[0] = impact
        lags = _structural_lag_matrices(A, n, p)
        for h in range(1, horizon):
            for lag in range(1, min(p, h) + 1):
                responses[h] += lags[lag - 1] @ responses[h - lag]
            if h == 1 and psi != 0.0:
                responses[h] += psi * impact
        return bool(
            np.all(np.sign(responses[:, restricted]) == signs[restricted][None, :])
        )

    n_draws = len(A_draws)
    if max_draws is None or int(max_draws) >= n_draws:
        candidate_indices = np.arange(n_draws, dtype=int)
    else:
        max_draws = int(max_draws)
        if max_draws < 1:
            raise ValueError("max_draws must be positive or None.")
        candidate_indices = np.unique(
            np.linspace(0, n_draws - 1, max_draws, dtype=int)
        )

    for d in candidate_indices:
        A = A_draws[d]
        Sigma = Sigma_draws[d]
        if stable_only and not _structural_is_stable(A, n, p):
            dropped_unstable += 1
            continue
        try:
            P = _safe_cholesky(0.5 * (Sigma + Sigma.T))
        except np.linalg.LinAlgError:
            dropped_cholesky += 1
            continue
        psi = _draw_psi(fit, d)
        accepted = None

        for attempt in range(1, max_tries + 1):
            K = rng.standard_normal((n, n))
            Q, R = np.linalg.qr(K)
            # Deterministic sign normalization of QR makes the Haar draw stable
            # across BLAS implementations without changing its distribution.
            dsign = np.sign(np.diag(R))
            dsign[dsign == 0] = 1.0
            Q = Q @ np.diag(dsign)
            candidate = P @ Q
            columns = range(n) if search_all_columns else (shock_slot,)

            for col in columns:
                impact = candidate[:, col]
                orientation = 1.0
                if signs_hold(A, impact, psi):
                    pass
                elif allow_flip and signs_hold(A, -impact, psi):
                    orientation = -1.0
                else:
                    continue

                candidate = candidate.copy()
                if orientation < 0:
                    candidate[:, col] *= -1.0
                    flipped += 1
                if col != shock_slot:
                    candidate[:, [shock_slot, col]] = candidate[:, [col, shock_slot]]
                accepted = candidate
                break
            if accepted is not None:
                accepted_matrices.append(accepted)
                used_indices.append(d)
                tries.append(attempt)
                break

        if accepted is None:
            dropped_max_tries += 1

    if not accepted_matrices:
        raise ValueError(
            "No posterior draw satisfied the sign restrictions. Relax the signs, "
            "raise max_tries, or inspect model stability."
        )

    tries = np.asarray(tries, dtype=int)
    n_used = len(accepted_matrices)
    n_candidates = len(candidate_indices)
    info = {
        "n_draws": n_draws,
        "n_candidates": n_candidates,
        "candidate_draw_indices": candidate_indices,
        "n_used": n_used,
        "used_draw_indices": np.asarray(used_indices, dtype=int),
        "shock_var": shock_var,
        "shock_name": shock_name or shock_var,
        "shock_index": shock_slot,
        "restrictions": dict(restrictions),
        "horizon": horizon,
        "tries": tries,
        "mean_tries": float(tries.mean()),
        "median_tries": float(np.median(tries)),
        "max_tries_observed": int(tries.max()),
        "flip_share": flipped / n_used,
        "dropped_unstable": dropped_unstable,
        "dropped_cholesky": dropped_cholesky,
        "dropped_max_tries": dropped_max_tries,
        "dropped_unstable_share": dropped_unstable / n_candidates,
        "dropped_cholesky_share": dropped_cholesky / n_candidates,
        "dropped_max_tries_share": dropped_max_tries / n_candidates,
        "search_all_columns": bool(search_all_columns),
        "identification": "QR sign restrictions",
    }
    return np.asarray(accepted_matrices), info


def impulse_responses(
    fit: Mapping[str, Any],
    *,
    shock_var: str,
    names: Sequence[str],
    p: Optional[int] = None,
    shock_size: float = 1.0,
    H_irf: int = 12,
    stable_only: bool = True,
    shock_unit: str = "std",
    A0_draws: Optional[np.ndarray] = None,
    A0_draw_indices: Optional[Sequence[int]] = None,
):
    """Posterior structural impulse responses with Cholesky or sign restrictions."""
    A_draws, Sigma_draws = _require_structural_draws(fit)
    names = list(names)
    n = Sigma_draws.shape[1]
    p = int(fit["p"] if p is None else p)
    if shock_var not in names:
        raise ValueError(f"{shock_var!r} not found in names.")
    j = names.index(shock_var)

    a0_map = None
    if A0_draws is not None:
        if A0_draw_indices is None:
            raise ValueError("A0_draw_indices is required with A0_draws.")
        A0_draws = np.asarray(A0_draws, dtype=float)
        indices = np.asarray(A0_draw_indices, dtype=int)
        if len(A0_draws) != len(indices):
            raise ValueError("A0_draws and A0_draw_indices must have equal length.")
        a0_map = {int(i): A0 for i, A0 in zip(indices, A0_draws)}

    out = []
    used = []
    dropped_unstable = 0
    for d, (A, Sigma) in enumerate(zip(A_draws, Sigma_draws)):
        if a0_map is not None and d not in a0_map:
            continue
        if stable_only and not _structural_is_stable(A, n, p):
            dropped_unstable += 1
            continue
        P = a0_map[d] if a0_map is not None else _safe_cholesky(0.5 * (Sigma + Sigma.T))
        psi = _draw_psi(fit, d)
        theta = _structural_irf_tensor(A, P, p, H_irf, psi=psi)
        impact = theta[:, :, j]
        if shock_unit == "std":
            response = shock_size * impact
        elif shock_unit == "level":
            denom = impact[0, j]
            if abs(denom) < 1e-12:
                continue
            response = impact * (shock_size / denom)
        else:
            raise ValueError("shock_unit must be 'std' or 'level'.")
        out.append(response)
        used.append(d)

    if not out:
        raise ValueError("No usable posterior draws remain for IRFs.")
    return np.asarray(out), {
        "n_draws": len(A_draws),
        "n_used": len(out),
        "used_draw_indices": np.asarray(used, dtype=int),
        "dropped_unstable": dropped_unstable,
        "shock_var": shock_var,
        "shock_unit": shock_unit,
        "identification": "sign restrictions" if a0_map is not None else "Cholesky",
        "ma1_included": bool(fit["config"].ma1),
    }


def forecast_error_variance_decomposition(
    fit: Mapping[str, Any],
    *,
    names: Sequence[str],
    p: Optional[int] = None,
    H_fevd: int = 12,
    stable_only: bool = True,
    A0_draws: Optional[np.ndarray] = None,
    A0_draw_indices: Optional[Sequence[int]] = None,
):
    """Posterior reference-state FEVD, including the optional MA(1) transfer.

    For constant-Gaussian models this is the standard FEVD.  For CSV/t models
    it deliberately reports the structural-dynamics FEVD normalized to the base
    innovation covariance Sigma.  The common scalar SV/t scale is not folded
    into horizon-specific predictive variance, which keeps the decomposition
    comparable across the seven Chan specifications.
    """
    A_draws, Sigma_draws = _require_structural_draws(fit)
    names = list(names)
    n = Sigma_draws.shape[1]
    p = int(fit["p"] if p is None else p)
    if len(names) != n:
        raise ValueError("names must contain one label per variable.")

    a0_map = None
    if A0_draws is not None:
        if A0_draw_indices is None:
            raise ValueError("A0_draw_indices is required with A0_draws.")
        idx = np.asarray(A0_draw_indices, dtype=int)
        A0_draws = np.asarray(A0_draws, dtype=float)
        a0_map = {int(i): A0 for i, A0 in zip(idx, A0_draws)}

    out = []
    used = []
    dropped_unstable = 0
    for d, (A, Sigma) in enumerate(zip(A_draws, Sigma_draws)):
        if a0_map is not None and d not in a0_map:
            continue
        if stable_only and not _structural_is_stable(A, n, p):
            dropped_unstable += 1
            continue
        P = a0_map[d] if a0_map is not None else _safe_cholesky(0.5 * (Sigma + Sigma.T))
        theta = _structural_irf_tensor(A, P, p, H_fevd, psi=_draw_psi(fit, d))
        cum = np.cumsum(theta**2, axis=0)
        total = cum.sum(axis=2, keepdims=True)
        fevd = np.divide(cum, total, out=np.zeros_like(cum), where=total > 0)
        out.append(fevd)
        used.append(d)

    if not out:
        raise ValueError("No usable posterior draws remain for FEVD.")
    return np.asarray(out), {
        "n_draws": len(A_draws),
        "n_used": len(out),
        "used_draw_indices": np.asarray(used, dtype=int),
        "dropped_unstable": dropped_unstable,
        "identification": "sign restrictions" if a0_map is not None else "Cholesky",
        "ma1_included": bool(fit["config"].ma1),
        "normalization": "base Sigma; common CSV/t scale excluded from horizon weighting",
        "axes": ("posterior_draw", "horizon", "response", "shock"),
    }


def historical_decomposition(
    fit: Mapping[str, Any],
    Y0: np.ndarray,
    shortYt: np.ndarray,
    *,
    names: Sequence[str],
    p: Optional[int] = None,
    stable_only: bool = True,
    reconstruction_tol: float = 1e-8,
    A0_draws: Optional[np.ndarray] = None,
    A0_draw_indices: Optional[Sequence[int]] = None,
):
    """Exact posterior historical decomposition of the fitted sample.

    For BVAR-CSV-t-MA the reduced-form residual is first inverted through the
    sampled MA filter, ``u_t = e_t + psi e_{t-1}``.  Each ``e_t`` is then
    decomposed through the draw-specific structural impact matrix.  CSV and
    Student-t scales are already embodied in the realized innovations, so no
    extra rescaling is required for the reconstruction identity.
    """
    A_draws, Sigma_draws = _require_structural_draws(fit)
    Y0 = np.asarray(Y0, dtype=float)
    Y = np.asarray(shortYt, dtype=float)
    names = list(names)
    Tt, n = Y.shape
    p = int(fit["p"] if p is None else p)
    if len(names) != n:
        raise ValueError("names must contain one label per variable.")
    Z = build_var_design(Y0, Y, p=p)
    initial_lags = [
        Z[0, 1 + lag * n:1 + (lag + 1) * n].copy()
        for lag in range(p)
    ]

    a0_map = None
    if A0_draws is not None:
        if A0_draw_indices is None:
            raise ValueError("A0_draw_indices is required with A0_draws.")
        idx = np.asarray(A0_draw_indices, dtype=int)
        A0_draws = np.asarray(A0_draws, dtype=float)
        a0_map = {int(i): A0 for i, A0 in zip(idx, A0_draws)}

    contributions_out = []
    base_out = []
    shocks_out = []
    reconstructed_out = []
    used = []
    max_errors = []
    dropped_unstable = 0

    for d, (A, Sigma) in enumerate(zip(A_draws, Sigma_draws)):
        if a0_map is not None and d not in a0_map:
            continue
        if stable_only and not _structural_is_stable(A, n, p):
            dropped_unstable += 1
            continue
        P = a0_map[d] if a0_map is not None else _safe_cholesky(0.5 * (Sigma + Sigma.T))
        psi = _draw_psi(fit, d)
        lags = _structural_lag_matrices(A, n, p)

        U = Y - Z @ A
        E = _ma_inverse_filter(U, psi) if fit["config"].ma1 else U
        structural_shocks = linalg.solve(P, E.T, assume_a="gen").T

        contributions = np.zeros((Tt, n, n), dtype=float)
        base = np.zeros((Tt, n), dtype=float)

        for t in range(Tt):
            # Current e_t contribution, one column per structural shock.
            contributions[t] = P * structural_shocks[t][None, :]
            # MA(1) adds psi * e_{t-1} directly to u_t.
            if fit["config"].ma1 and t > 0:
                contributions[t] += psi * (P * structural_shocks[t - 1][None, :])

            base[t] = A[0]
            for lag, A_lag in enumerate(lags, start=1):
                if t - lag >= 0:
                    contributions[t] += A_lag @ contributions[t - lag]
                    base[t] += A_lag @ base[t - lag]
                else:
                    initial_index = lag - t - 1
                    base[t] += A_lag @ initial_lags[initial_index]

        reconstructed = base + contributions.sum(axis=2)
        max_error = float(np.max(np.abs(reconstructed - Y)))
        if max_error > reconstruction_tol:
            raise ValueError(
                f"Historical decomposition failed for draw {d}; maximum "
                f"reconstruction error={max_error:.3e}."
            )

        contributions_out.append(contributions)
        base_out.append(base)
        shocks_out.append(structural_shocks)
        reconstructed_out.append(reconstructed)
        used.append(d)
        max_errors.append(max_error)

    if not contributions_out:
        raise ValueError("No usable posterior draws remain for historical decomposition.")

    hd = {
        "contributions": np.asarray(contributions_out),
        "base": np.asarray(base_out),
        "structural_shocks": np.asarray(shocks_out),
        "reconstructed": np.asarray(reconstructed_out),
        "observed": Y,
        "dates": np.arange(Tt),
        "names": names,
    }
    info = {
        "n_draws": len(A_draws),
        "n_used": len(contributions_out),
        "used_draw_indices": np.asarray(used, dtype=int),
        "dropped_unstable": dropped_unstable,
        "max_reconstruction_error": float(max(max_errors)),
        "identification": "sign restrictions" if a0_map is not None else "Cholesky",
        "ma1_included": bool(fit["config"].ma1),
        "axes": ("posterior_draw", "time", "response", "shock"),
    }
    return hd, info

def estimate_bvar_small(
    Y0,
    shortYt,
    p: int = 4,
    c1: float = 0.2**2,
    c2: float = 0.1**2,
    c3: float = 100.0,
    variable_names=CORE_LABELS,
):
    """Analytic four-variable Minnesota posterior from the previous version."""
    _, n = shortYt.shape
    k = 1 + n * p
    prior = prior_minnesota(Y0, shortYt, p, c1, c2, c3)
    Z = build_var_design(Y0, shortYt, p)
    A_mean = np.zeros((k, n))
    A_sd = np.zeros((k, n))
    chols = []

    for i in range(n):
        sl = slice(i * k, (i + 1) * k)
        V_i = prior["variance"][sl]
        beta_i = prior["mean"][sl]
        K_i = np.diag(1 / V_i) + (Z.T @ Z) / prior["sig2"][i]
        rhs_i = beta_i / V_i + (Z.T @ shortYt[:, i]) / prior["sig2"][i]
        C_i = _safe_cholesky(K_i)
        A_mean[:, i] = linalg.cho_solve((C_i, True), rhs_i)
        K_inv = linalg.cho_solve((C_i, True), np.eye(k))
        A_sd[:, i] = np.sqrt(np.diag(K_inv))
        chols.append(C_i)

    rows = []
    names = coefficient_names(variable_names, p)
    for i, equation in enumerate(variable_names):
        for j, coefficient in enumerate(names):
            rows.append({
                "equation": equation,
                "coefficient": coefficient,
                "posterior mean": A_mean[j, i],
                "posterior sd": A_sd[j, i],
            })

    return {
        "A_mean": A_mean,
        "A_sd": A_sd,
        "chols": chols,
        "sig2": prior["sig2"],
        "Sigma": np.diag(prior["sig2"]),
        "prior": prior,
        "summary": pd.DataFrame(rows),
        "p": p,
        "n": n,
        "k": k,
        "variables": list(variable_names),
    }


def draw_A(model: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    A = np.zeros_like(model["A_mean"])
    for i, C in enumerate(model["chols"]):
        A[:, i] = model["A_mean"][:, i] + linalg.solve_triangular(
            C.T, rng.standard_normal(model["k"]), lower=False
        )
    return A


def draw_bvar_small_posterior(model, nsims: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    return np.stack([draw_A(model, rng) for _ in range(nsims)], axis=0)


def companion_matrix(A: np.ndarray, n: int, p: int) -> np.ndarray:
    top = np.hstack([
        A[1 + lag * n:1 + (lag + 1) * n, :].T
        for lag in range(p)
    ])
    if p == 1:
        return top
    bottom = np.hstack([
        np.eye(n * (p - 1)),
        np.zeros((n * (p - 1), n)),
    ])
    return np.vstack([top, bottom])


def posterior_diagnostics(draws: np.ndarray, p: int = 4):
    n = draws.shape[2]
    spectral_radius = np.array([
        np.max(np.abs(np.linalg.eigvals(companion_matrix(A, n, p))))
        for A in draws
    ])
    return {
        "spectral_radius": spectral_radius,
        "unstable_share": float(np.mean(spectral_radius >= 1)),
    }


def forecast_bvar_small(
    Y0,
    shortYt,
    data_tpk,
    is_last_miss,
    t,
    T,
    p: int = 4,
    nsims: int = 5000,
    burnin: int = 100,
    rng: Optional[np.random.Generator] = None,
):
    """Legacy four-variable BVAR-small forecast using the generalized helpers."""
    if rng is None:
        rng = np.random.default_rng()
    config = BVARConfig(
        name="BVAR-small",
        prior="minnesota",
        stochastic_volatility=False,
        student_t=False,
        ma1=False,
        p=p,
        c1=0.2**2,
        c2=0.1**2,
        c3=100.0,
    )
    n = shortYt.shape[1]
    tmp0 = np.zeros((nsims, 2 * n + 1))
    tmp1 = np.zeros((nsims, 2 * n + 1))
    for s, state in enumerate(_iter_minnesota_draws(Y0, shortYt, config, nsims, burnin, rng)):
        forecasts = _forecast_from_state(
            state, shortYt, data_tpk, is_last_miss, t, T, config, rng
        )
        mean0, lden0, joint0, _ = forecasts[0]
        tmp0[s] = np.concatenate([mean0, lden0, [joint0]])
        mean1, lden1, joint1, valid1 = forecasts[1]
        if valid1:
            tmp1[s] = np.concatenate([mean1, lden1, [joint1]])
    return tmp0, tmp1


def run_bvar_small(
    rt_data,
    nonrev_data,
    tcode,
    var_type,
    p: int = 4,
    T0: int = 41,
    T: int = 208,
    nsims: int = 5000,
    burnin: int = 100,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n = len(CORE_VARIABLES)
    yhat0 = np.zeros((T - T0 + 1, 3 * n + 1))
    yhat1 = np.zeros((T - 1 - T0 + 1, 3 * n + 1))

    for t in range(T0, T + 1):
        data_t, data_tpk = loaddata(rt_data, nonrev_data, t, T0, tcode, var_type)
        data_t, is_last_miss = trim_data(data_t)
        small = data_t[:, CORE_VARIABLES]
        Y0 = small[:p]
        shortYt = small[p:]
        targets = data_tpk[:, CORE_VARIABLES]
        tmp0, tmp1 = forecast_bvar_small(
            Y0, shortYt, targets, is_last_miss, t, T,
            p=p, nsims=nsims, burnin=burnin, rng=rng,
        )
        point0, lpl0 = aggregate_forecast_draws(tmp0, n)
        yhat0[t - T0] = np.concatenate([targets[0], point0, lpl0])
        if t <= T - 1:
            point1, lpl1 = aggregate_forecast_draws(tmp1, n)
            yhat1[t - T0] = np.concatenate([targets[1], point1, lpl1])

    obs0, fc0 = yhat0[4:, :n], yhat0[4:, n:2*n]
    obs1, fc1 = yhat1[3:, :n], yhat1[3:, n:2*n]
    rmsfe = np.column_stack([
        np.sqrt(np.mean((obs0 - fc0) ** 2, axis=0)),
        np.sqrt(np.mean((obs1 - fc1) ** 2, axis=0)),
    ])
    alpl = np.column_stack([
        yhat0[4:, 2*n:3*n].mean(axis=0),
        yhat1[3:, 2*n:3*n].mean(axis=0),
    ])
    return {
        "model_name": "BVAR-small",
        "variables": CORE_LABELS.copy(),
        "evaluation_indices": list(range(n)),
        "rmsfe": rmsfe,
        "alpl": alpl,
        "yhat0": yhat0,
        "yhat1": yhat1,
        "n": n,
        "p": p,
        "k": 1 + n * p,
        "T0": T0,
        "T": T,
        "nsims": nsims,
        "burnin": burnin,
        "seed": seed,
    }
