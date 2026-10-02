"""Plotting helpers for the generalized Chan (2020) large-BVAR notebook."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_VARIABLES = [0, 6, 7, 11]
CORE_LABELS = [
    "GDP growth",
    "Industrial production growth",
    "Unemployment rate",
    "PCE inflation",
]


def _axes_grid(n_panels: int, ncols: int = 2, width: float = 5.5, height: float = 3.2):
    ncols = min(ncols, max(1, n_panels))
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes


def plot_model_series(sample, indices: Optional[Sequence[int]] = None, labels: Optional[Sequence[str]] = None):
    """Plot selected transformed series entering a BVAR sample."""
    data = np.asarray(sample["data"], dtype=float)
    all_names = list(sample.get("variables", [f"var{i+1}" for i in range(data.shape[1])]))

    if indices is None:
        # The large-BVAR notebook should show the full information set.
        indices = list(range(data.shape[1]))
        labels = labels or all_names
    else:
        indices = list(indices)
        labels = list(labels or [all_names[i] for i in indices])

    fig, axes = _axes_grid(
        len(indices),
        ncols=4 if len(indices) > 8 else 2,
        width=4.2,
        height=2.7,
    )
    for ax, idx, name in zip(axes, indices, labels):
        ax.plot(data[:, idx], lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Model observation")
    fig.tight_layout()
    return fig


def plot_residuals(ols, indices: Optional[Sequence[int]] = None):
    """OLS residuals used to inspect scale and remaining dynamics."""
    residuals = np.asarray(ols["residuals"], dtype=float)
    names = list(ols["B_table"].columns)
    if indices is None:
        indices = list(range(residuals.shape[1]))
    indices = list(indices)
    fig, axes = _axes_grid(len(indices), ncols=2)
    for ax, idx in zip(axes, indices):
        ax.plot(residuals[:, idx], lw=0.8)
        ax.axhline(0, lw=0.8)
        ax.set_title(f"{names[idx]} residual")
    fig.tight_layout()
    return fig


def plot_minnesota_prior(prior, variable_names):
    """Minnesota prior standard deviations for all coefficients/equations."""
    n = prior["n"]
    k = prior["k"]
    prior_sd = np.sqrt(prior["variance"].reshape((k, n), order="F"))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.18 * k)))
    image = ax.imshow(np.log10(prior_sd), aspect="auto")
    ax.set_yticks(
        np.arange(k),
        ["const"]
        + [
            f"L{lag} {name}"
            for lag in range(1, prior["p"] + 1)
            for name in variable_names
        ],
    )
    ax.set_xticks(np.arange(n), variable_names, rotation=30, ha="right")
    ax.set_title("Minnesota prior standard deviations (log10 scale)")
    ax.set_xlabel("Equation")
    ax.set_ylabel("Coefficient")
    fig.colorbar(image, ax=ax, label="log10 prior sd")
    fig.tight_layout()
    return fig


def plot_posterior_marginals(draws, model):
    """Legacy BVAR-small marginal posterior plot."""
    names = model["variables"]
    n = model["n"]
    fig, axes = _axes_grid(n, ncols=2)
    for i, (ax, name) in enumerate(zip(axes, names)):
        regressor = 1 + i
        values = draws[:, regressor, i]
        ax.hist(values, bins=35, density=True, alpha=0.8)
        ax.axvline(model["A_mean"][regressor, i], lw=1.2, label="posterior mean")
        ax.axvline(0, lw=1.0, linestyle="--", label="prior mean")
        ax.set_title(f"{name}: own L1")
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot_shrinkage(ols, model):
    """Legacy OLS coefficients against BVAR-small posterior means."""
    x = np.asarray(ols["B"], dtype=float).reshape(-1)
    y = np.asarray(model["A_mean"], dtype=float).reshape(-1)
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(x, y, s=18, alpha=0.7)
    ax.plot([low, high], [low, high], linestyle="--", lw=1.0)
    ax.axhline(0, lw=0.7)
    ax.axvline(0, lw=0.7)
    ax.set_xlabel("OLS coefficient")
    ax.set_ylabel("Posterior mean")
    ax.set_title("Shrinkage: posterior mean against OLS")
    fig.tight_layout()
    return fig


def plot_spectral_radius(diagnostics):
    """Legacy posterior distribution of the VAR spectral radius."""
    rho = np.asarray(diagnostics["spectral_radius"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rho, bins=40, density=True, alpha=0.8)
    ax.axvline(1.0, linestyle="--", lw=1.2, label="stability boundary")
    ax.set_xlabel("Spectral radius")
    ax.set_title("Posterior stability diagnostic")
    ax.legend()
    fig.tight_layout()
    return fig


def active_mcmc_parameters(fit) -> list[str]:
    """Return scalar parameters that are actually sampled in this model."""
    out = []
    for key in ("rho", "sigh2", "nu", "psi", "h_last", "gamma_share"):
        values = np.asarray(fit["draws"].get(key, []), dtype=float)
        if np.isfinite(values).any():
            out.append(key)
    return out


def plot_mcmc_traces(fit, parameters: Optional[Sequence[str]] = None):
    """Trace plots for the active CSV/t/MA or SSVS latent parameters."""
    parameters = list(parameters or active_mcmc_parameters(fit))
    if not parameters:
        raise ValueError("This model has no scalar MCMC latent parameters to plot.")

    fig, axes = _axes_grid(len(parameters), ncols=1, width=9.0, height=2.4)
    for ax, parameter in zip(axes, parameters):
        values = np.asarray(fit["draws"][parameter], dtype=float)
        ax.plot(values, lw=0.7)
        ax.axhline(np.nanmean(values), lw=1.0, linestyle="--", label="posterior mean")
        ax.set_ylabel(parameter)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Saved draw")
    fig.suptitle(f"{fit['model_name']} — MCMC traces")
    fig.tight_layout()
    return fig


def plot_mcmc_marginals(fit, parameters: Optional[Sequence[str]] = None):
    """Marginal posterior histograms for active scalar latent parameters."""
    parameters = list(parameters or active_mcmc_parameters(fit))
    if not parameters:
        raise ValueError("This model has no scalar MCMC latent parameters to plot.")

    fig, axes = _axes_grid(len(parameters), ncols=2)
    for ax, parameter in zip(axes, parameters):
        values = np.asarray(fit["draws"][parameter], dtype=float)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=35, density=True, alpha=0.8)
        ax.axvline(values.mean(), lw=1.1, linestyle="--")
        ax.set_title(parameter)
    fig.suptitle(f"{fit['model_name']} — posterior marginals")
    fig.tight_layout()
    return fig


def plot_sigma_diagonal(fit, indices: Optional[Sequence[int]] = None, labels: Optional[Sequence[str]] = None):
    """Posterior distributions of selected diagonal elements of Sigma."""
    values = np.asarray(fit["draws"]["sigma_diag"], dtype=float)
    if indices is None:
        indices = CORE_VARIABLES if values.shape[1] >= 12 else range(values.shape[1])
    indices = list(indices)
    labels = list(labels or (CORE_LABELS if indices == CORE_VARIABLES else [f"var{i+1}" for i in indices]))

    fig, axes = _axes_grid(len(indices), ncols=2)
    for ax, idx, label in zip(axes, indices, labels):
        ax.hist(values[:, idx], bins=35, density=True, alpha=0.8)
        ax.axvline(np.mean(values[:, idx]), lw=1.0, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel(r"$\Sigma_{ii}$")
    fig.suptitle(f"{fit['model_name']} — covariance diagonal")
    fig.tight_layout()
    return fig



# ----------------------------------------------------------------------
# MCMC diagnosis and sign-identification strength
# ----------------------------------------------------------------------

def plot_mcmc_diagnostic_traces(diagnostics, ncols=2):
    """Compact trace catalogue produced by bvar_mcmc_diagnostics()."""
    if not diagnostics.get("is_mcmc", False):
        raise ValueError("This specification uses direct/iid posterior draws, not an MCMC chain.")

    series = diagnostics["series"]
    names = list(series)
    if not names:
        raise ValueError("No diagnostic chains are available.")

    fig, axes = _axes_grid(
        len(names),
        ncols=ncols,
        width=5.8,
        height=2.35,
    )
    for ax, name in zip(axes, names):
        values = np.asarray(series[name], dtype=float)
        ax.plot(values, lw=0.55)
        ax.axhline(np.nanmean(values), linestyle="--", lw=0.9)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("retained Gibbs/MH draw")
    fig.suptitle(f"{diagnostics['model_name']} — MCMC trace diagnosis")
    fig.tight_layout()
    return fig


def plot_mcmc_diagnostic_acf(diagnostics, max_lag=None, ncols=2):
    """Autocorrelation plots for the same chains used in the ESS/MCSE table."""
    if not diagnostics.get("is_mcmc", False):
        raise ValueError("This specification uses direct/iid posterior draws, not an MCMC chain.")

    acf = diagnostics["acf"]
    names = list(acf)
    if not names:
        raise ValueError("No ACF chains are available.")

    fig, axes = _axes_grid(
        len(names),
        ncols=ncols,
        width=5.8,
        height=2.35,
    )
    for ax, name in zip(axes, names):
        values = np.asarray(acf[name], dtype=float)
        if max_lag is not None:
            values = values[: int(max_lag) + 1]
        lags = np.arange(len(values))
        ax.vlines(lags, 0.0, values, lw=0.9)
        ax.axhline(0.0, lw=0.7)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF")
    fig.suptitle(f"{diagnostics['model_name']} — MCMC autocorrelation")
    fig.tight_layout()
    return fig


def plot_sign_identification(diags, labels=None, bins=25, show=True):
    """Volume + angular-dispersion diagnosis for sign restrictions.

    Adapted from the user's existing structural-diagnostics plotting layer.
    The left panel measures the admissible-set volume; the right panel measures
    dispersion of admissible structural directions.  The dashed right-panel
    reference is the normalisation-only baseline when available.
    """
    if isinstance(diags, dict):
        diags = [diags]

    if labels is None:
        labels = []
        for d in diags:
            restriction_label = ", ".join(
                f"{name}{'+' if sign > 0 else '−'}"
                for name, sign in d["restrictions"].items()
                if sign != 0
            )
            if d.get("horizon", 1) > 1:
                restriction_label += f", h={d['horizon']}"
            labels.append(restriction_label)

    if len(labels) != len(diags):
        raise ValueError("labels must have one entry per diagnostic.")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    for d, label in zip(diags, labels):
        axes[0].hist(
            d["acceptance"],
            bins=bins,
            range=(0, 1),
            density=True,
            alpha=0.45,
            label=label,
        )
        axes[1].hist(
            d["aperture90"],
            bins=bins,
            range=(0, 90),
            density=True,
            alpha=0.45,
        )

    baseline = next(
        (d.get("baseline") for d in diags if d.get("baseline") is not None),
        None,
    )
    if baseline is not None:
        axes[1].axvline(
            np.median(baseline["aperture90"]),
            lw=1.2,
            linestyle="--",
            label="normalisation-only floor",
        )
        axes[1].legend(frameon=False, fontsize=7)

    axes[0].set_xlabel(r"acceptance rate $a_d$ (per posterior draw)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Size of the admissible set (volume)")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].set_xlabel(r"90% angular aperture $A_{90,d}$ (degrees)")
    axes[1].set_title("Dispersion of admissible directions")
    axes[1].set_xlim(0, 90)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    model_title = diags[0].get("model_name", "BVAR")
    fig.suptitle(f"{model_title} — sign-restriction identification strength", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    summary = pd.DataFrame(
        {
            "restriction set": labels,
            "median acceptance a_d": [
                np.median(d["acceptance"]) for d in diags
            ],
            "median A90 (deg)": [
                np.median(d["aperture90"]) for d in diags
            ],
            "median R_d": [
                np.median(d["R"]) for d in diags
            ],
            "median vMF kappa": [
                np.median(d["kappa"]) for d in diags
            ],
            "draws measured": [
                len(d["R"]) for d in diags
            ],
        }
    ).set_index("restriction set").round(3)

    if show:
        plt.show()

    return fig, summary


def plot_rmsfe(result):
    """RMSFE for the four evaluation variables."""
    x = np.arange(len(result["variables"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, result["rmsfe"][:, 0], width, label="Nowcast")
    ax.bar(x + width / 2, result["rmsfe"][:, 1], width, label="1-quarter-ahead")
    ax.set_xticks(x, result["variables"], rotation=15, ha="right")
    ax.set_ylabel("RMSFE")
    ax.set_title(f"{result['model_name']} — RMSFE")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_alpl(result):
    """Average log predictive likelihood for the four evaluation variables."""
    x = np.arange(len(result["variables"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, result["alpl"][:, 0], width, label="Nowcast")
    ax.bar(x + width / 2, result["alpl"][:, 1], width, label="1-quarter-ahead")
    ax.set_xticks(x, result["variables"], rotation=15, ha="right")
    ax.set_ylabel("ALPL")
    ax.set_title(f"{result['model_name']} — ALPL")
    ax.legend()
    fig.tight_layout()
    return fig


def _evaluation_block(result, horizon: int):
    n_model = int(result["n"])
    indices = list(result.get("evaluation_indices", range(len(result["variables"]))))
    if horizon == 0:
        block = result["yhat0"][4:]
        label = "Nowcast"
    elif horizon == 1:
        block = result["yhat1"][3:]
        label = "1-quarter-ahead"
    else:
        raise ValueError("horizon must be 0 or 1")
    observed = block[:, indices]
    forecast = block[:, n_model + np.asarray(indices)]
    return block, observed, forecast, label


def plot_forecasts(result, horizon: int = 0):
    """Observed versus recursive point forecasts for the evaluation variables."""
    _, observed, forecast, horizon_label = _evaluation_block(result, horizon)
    dates = pd.period_range("1975Q1", periods=len(observed), freq="Q").to_timestamp()
    fig, axes = _axes_grid(len(result["variables"]), ncols=2)
    for i, (ax, name) in enumerate(zip(axes, result["variables"])):
        ax.plot(dates, observed[:, i], lw=1.0, label="Observed")
        ax.plot(dates, forecast[:, i], lw=1.0, label="Forecast")
        ax.set_title(name)
    axes[0].legend()
    fig.suptitle(f"{result['model_name']} — {horizon_label}")
    fig.tight_layout()
    return fig


def plot_forecast_errors(result, horizon: int = 0):
    """Recursive forecast errors for the evaluation variables."""
    _, observed, forecast, horizon_label = _evaluation_block(result, horizon)
    errors = observed - forecast
    dates = pd.period_range("1975Q1", periods=len(errors), freq="Q").to_timestamp()
    fig, axes = _axes_grid(len(result["variables"]), ncols=2)
    for i, (ax, name) in enumerate(zip(axes, result["variables"])):
        ax.plot(dates, errors[:, i], lw=0.9)
        ax.axhline(0, lw=0.8)
        ax.set_title(name)
    fig.suptitle(f"{result['model_name']} forecast errors — {horizon_label}")
    fig.tight_layout()
    return fig


def plot_model_comparison(comparison: pd.DataFrame, metric: str = "RMSFE", horizon: str = "Nowcast"):
    """Compare completed recursive model runs variable by variable."""
    if metric not in {"RMSFE", "ALPL"}:
        raise ValueError("metric must be 'RMSFE' or 'ALPL'")
    subset = comparison.loc[comparison["horizon"] == horizon].copy()
    variables = list(dict.fromkeys(subset["variable"]))
    models = list(dict.fromkeys(subset["model"]))
    x = np.arange(len(variables))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(11, 5))
    for j, model in enumerate(models):
        vals = (
            subset.loc[subset["model"] == model]
            .set_index("variable")
            .reindex(variables)[metric]
            .to_numpy()
        )
        offset = (j - (len(models) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=model)
    ax.set_xticks(x, variables, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison — {metric}, {horizon}")
    ax.legend()
    fig.tight_layout()
    return fig


def display_recursive_suite(suite, detail_model: Optional[str] = None):
    """Compact notebook display for ``run_large_bvar_suite`` output."""
    from IPython.display import Markdown, display

    print("Evaluation: lower RMSFE is better; higher ALPL is better.")
    for title, table in suite["tables"].items():
        print(f"\
{title}")
        display(table.round(4))

    print("\
Best model by evaluation task")
    display(suite["best_by_task"].round(4))
    for comment in suite["comments"]:
        display(Markdown(comment))

    comparison = suite["comparison"]
    for metric in ("RMSFE", "ALPL"):
        for horizon in ("Nowcast", "1-quarter-ahead"):
            plot_model_comparison(comparison, metric=metric, horizon=horizon)
            plt.show()

    if detail_model is not None:
        replications = suite["replications"]
        if detail_model not in replications:
            raise KeyError(
                f"detail_model={detail_model!r} is not one of {list(replications)}"
            )
        result = replications[detail_model]
        print(f"\
Detailed recursive paths — {detail_model}")
        plot_forecasts(result, horizon=0); plt.show()
        plot_forecasts(result, horizon=1); plt.show()
        plot_forecast_errors(result, horizon=0); plt.show()
        plot_forecast_errors(result, horizon=1); plt.show()

    return suite


# -----------------------------------------------------------------------------
# Large-BVAR posterior coefficients, predictive fans and structural analysis
# -----------------------------------------------------------------------------

def _fit_variable_names(fit):
    n = int(fit["n"])
    names = list(fit.get("variables", []))
    return names if len(names) == n else [f"var{i + 1}" for i in range(n)]


def plot_coefficient_traces(fit, indices=None):
    """Trace plots of selected own first-lag coefficients.

    This is the fallback MCMC view for models such as BVAR-Minn/NCP/IP that do
    not have scalar latent states like rho, nu or psi.
    """
    A = fit.get("draws", {}).get("A")
    if A is None:
        raise ValueError("Coefficient draws were not stored. Use store_A=True.")
    A = np.asarray(A, dtype=float)
    names = _fit_variable_names(fit)
    n = len(names)
    if indices is None:
        indices = CORE_VARIABLES if n >= 12 else range(n)
    indices = list(indices)
    fig, axes = _axes_grid(len(indices), ncols=1, width=9.0, height=2.3)
    for ax, i in zip(axes, indices):
        values = A[:, 1 + i, i]
        ax.plot(values, lw=0.65)
        ax.axhline(values.mean(), linestyle="--", lw=1.0)
        ax.set_ylabel(f"{names[i]} own L1")
    axes[-1].set_xlabel("Saved draw")
    fig.suptitle(f"{fit['model_name']} — coefficient traces")
    fig.tight_layout()
    return fig


def plot_coefficient_marginals(fit, indices=None):
    """Marginal posterior histograms of selected own first-lag coefficients."""
    A = fit.get("draws", {}).get("A")
    if A is None:
        raise ValueError("Coefficient draws were not stored. Use store_A=True.")
    A = np.asarray(A, dtype=float)
    names = _fit_variable_names(fit)
    n = len(names)
    if indices is None:
        indices = CORE_VARIABLES if n >= 12 else range(n)
    indices = list(indices)
    fig, axes = _axes_grid(len(indices), ncols=2)
    for ax, i in zip(axes, indices):
        values = A[:, 1 + i, i]
        ax.hist(values, bins=35, density=True, alpha=0.8)
        ax.axvline(values.mean(), linestyle="--", lw=1.0)
        ax.set_title(f"{names[i]}: own L1")
    fig.suptitle(f"{fit['model_name']} — coefficient marginals")
    fig.tight_layout()
    return fig


def plot_forecast_fan(forecast, indices=None, last_obs=72, quantiles=(10, 25, 50, 75, 90)):
    """Posterior predictive fan chart from ``forecast_bvar(...)["fan"]``.

    The same dictionary is also returned by the low-level
    ``forecast_paths_from_fit`` helper. By default every fitted variable is shown.
    The bands are P10-P90 and
    P25-P75 with the posterior median in the center.
    """
    paths = np.asarray(forecast["paths"], dtype=float)
    history = np.asarray(forecast["history"], dtype=float)
    names = list(forecast["variables"])
    H = int(forecast["H"])
    n = paths.shape[2]
    if indices is None:
        indices = list(range(n))
    else:
        indices = list(indices)
    q = np.percentile(paths[:, :, indices], quantiles, axis=0)
    x_hist = np.arange(len(history))
    x_fc = np.arange(len(history) - 1, len(history) + H)

    fig, axes = _axes_grid(
        len(indices),
        ncols=4 if len(indices) > 8 else 2,
        width=4.2,
        height=2.8,
    )
    for panel, (ax, idx) in enumerate(zip(axes, indices)):
        last = history[-1, idx]
        ax.plot(x_hist[-last_obs:], history[-last_obs:, idx], lw=1.0, label="Observed")
        ax.fill_between(x_fc, np.r_[last, q[0, :, panel]], np.r_[last, q[4, :, panel]], alpha=0.20, label="P10-P90")
        ax.fill_between(x_fc, np.r_[last, q[1, :, panel]], np.r_[last, q[3, :, panel]], alpha=0.35, label="P25-P75")
        ax.plot(x_fc, np.r_[last, q[2, :, panel]], lw=1.3, label="Median")
        ax.axvline(len(history) - 1, linestyle=":", lw=0.8)
        ax.set_title(names[idx])
        ax.set_xlabel("Model observation / forecast horizon")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{forecast['model_name']} — posterior predictive fan, H={H}")
    fig.tight_layout()
    return fig


def plot_irf_grid(irfs, names, shock_label, quantiles=(16, 50, 84), method="Sign restrictions", indices=None):
    """Posterior IRF grid with median and pointwise uncertainty bands."""
    irfs = np.asarray(irfs, dtype=float)
    names = list(names)
    n = irfs.shape[2]
    if indices is None:
        indices = list(range(n))
    else:
        indices = list(indices)
    q = np.percentile(irfs[:, :, indices], quantiles, axis=0)
    fig, axes = _axes_grid(
        len(indices),
        ncols=4 if len(indices) > 8 else 2,
        width=4.2,
        height=2.8,
    )
    horizons = np.arange(irfs.shape[1])
    for panel, (ax, idx) in enumerate(zip(axes, indices)):
        ax.fill_between(horizons, q[0, :, panel], q[2, :, panel], alpha=0.25)
        ax.plot(horizons, q[1, :, panel], lw=1.3, label="Median")
        ax.axhline(0.0, lw=0.8)
        ax.set_title(f"Response of {names[idx]}")
        ax.set_xlabel("Horizon")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Responses to {shock_label} — {method}")
    fig.tight_layout()
    return fig


def plot_fevd(fevd_draws, names, summary="mean", response_indices=None, ncols_subplot=4):
    """Stacked FEVD panels for selected response variables."""
    fevd_draws = np.asarray(fevd_draws, dtype=float)
    names = list(names)
    if fevd_draws.ndim != 4:
        raise ValueError("fevd_draws must have shape (draw, horizon, response, shock).")
    if summary == "mean":
        fevd = fevd_draws.mean(axis=0)
    elif summary == "median":
        fevd = np.median(fevd_draws, axis=0)
        fevd /= fevd.sum(axis=-1, keepdims=True)
    else:
        raise ValueError("summary must be 'mean' or 'median'.")
    H, n, n_shocks = fevd.shape
    if response_indices is None:
        response_indices = list(range(n))
    else:
        response_indices = list(response_indices)
    fig, axes = _axes_grid(
        len(response_indices),
        ncols=ncols_subplot if len(response_indices) > ncols_subplot else min(2, len(response_indices)),
        width=4.4,
        height=3.0,
    )
    horizons = np.arange(1, H + 1)
    for ax, idx in zip(axes, response_indices):
        ax.stackplot(horizons, fevd[:, idx, :].T, labels=names[:n_shocks])
        ax.set_ylim(0, 1)
        ax.set_xlim(1, H)
        ax.set_title(names[idx])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Variance share")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Structural shock", loc="lower center", ncol=min(5, len(labels)), bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Forecast Error Variance Decomposition")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    return fig


def plot_historical_decomposition(hd, names, response_vars=None, summary="mean", include_base=False, max_obs=120):
    """Posterior historical decomposition with positive/negative stacked bars."""
    contributions = np.asarray(hd["contributions"], dtype=float)
    base = np.asarray(hd["base"], dtype=float)
    observed = np.asarray(hd["observed"], dtype=float)
    names = list(names)
    if summary == "mean":
        C = contributions.mean(axis=0)
        B = base.mean(axis=0)
    elif summary == "median":
        C = np.median(contributions, axis=0)
        B = np.median(base, axis=0)
    else:
        raise ValueError("summary must be 'mean' or 'median'.")
    if response_vars is None:
        selected = list(names)
    elif isinstance(response_vars, str):
        selected = [response_vars]
    else:
        selected = list(response_vars)

    start = max(0, observed.shape[0] - int(max_obs))
    x = np.arange(start, observed.shape[0])
    fig, axes = _axes_grid(
        len(selected),
        ncols=4 if len(selected) > 8 else 2,
        width=4.5,
        height=3.2,
    )
    for ax, response in zip(axes, selected):
        i = names.index(response)
        components = C[start:, i, :]
        labels = [f"{name} shock" for name in names]
        if include_base:
            components = np.column_stack([components, B[start:, i]])
            labels.append("Base")
            target = observed[start:, i]
            target_label = "Observed"
        else:
            target = observed[start:, i] - B[start:, i]
            target_label = "Observed - base"

        pos_bottom = np.zeros(len(x))
        neg_bottom = np.zeros(len(x))
        for j, lab in enumerate(labels):
            vals = components[:, j]
            pos = np.where(vals > 0, vals, 0.0)
            neg = np.where(vals < 0, vals, 0.0)
            bars = ax.bar(x, pos, bottom=pos_bottom, width=0.9, label=lab)
            color = bars.patches[0].get_facecolor()
            ax.bar(x, neg, bottom=neg_bottom, width=0.9, color=color)
            pos_bottom += pos
            neg_bottom += neg
        ax.plot(x, target, lw=1.2, label=target_label)
        ax.axhline(0, lw=0.7)
        ax.set_title(response)
        ax.set_xlabel("Model observation")
        ax.set_ylabel("Contribution")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Historical component", loc="lower center", ncol=min(5, len(labels)), bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Historical Decomposition")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    return fig


def plot_sign_restriction_diagnostics(info):
    """Compact diagnostic of QR sign-restriction search effort."""
    tries = np.asarray(info["tries"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(tries, bins=min(40, max(10, int(np.sqrt(len(tries))))), alpha=0.8)
    ax.axvline(np.median(tries), linestyle="--", lw=1.1, label=f"median={np.median(tries):.1f}")
    ax.set_xlabel("QR rotations tried before acceptance")
    ax.set_ylabel("Posterior draws")
    ax.set_title(f"Sign restrictions — accepted {info['n_used']} / {info.get('n_candidates', info['n_draws'])} searched draws")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
