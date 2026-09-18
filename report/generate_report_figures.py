from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shielding_mc import (
    ShieldingModel,
    analog_tally_variance,
    analytical_transmission,
    biased_tally_variance,
    optimal_delta,
    simulate_analog,
    simulate_biased,
)

FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODEL = ShieldingModel(
    thickness=100.0,
    sigma_t=0.10,
    sigma_s=0.05,
    sigma_a=0.05,
)
N_REPLICATE = 50_000
N_REPLICATES = 20
DELTAS = np.round(np.arange(0.50, 1.001, 0.05), 2)
BASE_SEED = 10_000

T_EXACT = analytical_transmission(MODEL)
DELTA_OPT = optimal_delta(MODEL)
VAR_ANALOG = analog_tally_variance(MODEL)
VAR_OPT = biased_tally_variance(MODEL, DELTA_OPT)


def collect_replicates():
    rows = []
    for replicate in range(N_REPLICATES):
        analog = simulate_analog(
            MODEL,
            N_REPLICATE,
            seed=BASE_SEED + 10_000 * replicate,
        )
        rows.append({
            "algorithm": "analog",
            "delta": np.nan,
            "replicate": replicate,
            "estimate": analog.estimate,
            "reported_se": analog.standard_error,
            "tally_variance": analog.tally_variance,
            "cpu_time": analog.elapsed_cpu,
        })
        for j, delta in enumerate(DELTAS, start=1):
            result = simulate_biased(
                MODEL,
                N_REPLICATE,
                delta=float(delta),
                seed=BASE_SEED + 10_000 * replicate + j,
            )
            rows.append({
                "algorithm": "biased",
                "delta": float(delta),
                "replicate": replicate,
                "estimate": result.estimate,
                "reported_se": result.standard_error,
                "tally_variance": result.tally_variance,
                "cpu_time": result.elapsed_cpu,
            })
    return pd.DataFrame(rows)


def summarize(df):
    summaries = []
    groups = [("analog", np.nan, df[df["algorithm"] == "analog"])]
    groups += [
        ("biased", float(delta), group)
        for delta, group in df[df["algorithm"] == "biased"].groupby("delta")
    ]
    for algorithm, delta, group in groups:
        cpu_per_history = group["cpu_time"].median() / N_REPLICATE
        sample_var = group["tally_variance"].mean()
        theory_var = (
            VAR_ANALOG
            if algorithm == "analog"
            else biased_tally_variance(MODEL, delta)
        )
        summaries.append({
            "algorithm": algorithm,
            "delta": delta,
            "mean_estimate": group["estimate"].mean(),
            "bias_vs_exact": group["estimate"].mean() - T_EXACT,
            "between_run_sd": group["estimate"].std(ddof=1),
            "mean_reported_se": group["reported_se"].mean(),
            "sample_tally_variance": sample_var,
            "theory_tally_variance": theory_var,
            "median_cpu_per_history": cpu_per_history,
            "sample_efficiency": 1.0 / (cpu_per_history * sample_var),
            "theory_efficiency": 1.0 / (cpu_per_history * theory_var),
        })
    out = pd.DataFrame(summaries)
    analog = out[out["algorithm"] == "analog"].iloc[0]
    out["relative_efficiency_sample"] = out["sample_efficiency"] / analog["sample_efficiency"]
    out["relative_efficiency_theory"] = out["theory_efficiency"] / analog["theory_efficiency"]
    return out


replicates = collect_replicates()
summary = summarize(replicates)
biased = summary[summary["algorithm"] == "biased"].sort_values("delta").copy()
analog = summary[summary["algorithm"] == "analog"].iloc[0]

summary.to_csv(REPORT_DIR / "benchmark_summary.csv", index=False)

# Figure 1: exact variance landscape.
delta_dense = np.linspace(0.35, 1.0, 500)
variance_dense = np.array([biased_tally_variance(MODEL, d) for d in delta_dense])
plt.figure(figsize=(7.4, 4.6))
plt.plot(delta_dense, variance_dense, label="Varianza exacta del estimador sesgado")
plt.axvline(DELTA_OPT, linestyle="--", label=rf"$\delta_{{opt}}={DELTA_OPT:.3f}$")
plt.axhline(VAR_ANALOG, linestyle=":", label="Varianza del método análogo")
plt.xlabel(r"Factor de dilución, $\delta$")
plt.ylabel("Varianza del tally por historia")
plt.title("Efecto teórico del trajectory biasing sobre la varianza")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "varianza_teorica.png", dpi=180, bbox_inches="tight")
plt.close()

# Figure 2: exact vs empirical variance.
plt.figure(figsize=(7.4, 4.6))
plt.plot(biased["delta"], biased["theory_tally_variance"], marker="o", label="Varianza exacta")
plt.plot(biased["delta"], biased["sample_tally_variance"], marker="s", label="Monte Carlo repetido")
plt.axvline(DELTA_OPT, linestyle="--", label=rf"$\delta_{{opt}}={DELTA_OPT:.3f}$")
plt.axhline(VAR_ANALOG, linestyle=":", label="Método análogo")
plt.xlabel(r"Factor de dilución, $\delta$")
plt.ylabel("Varianza del tally por historia")
plt.title("Validación de la varianza teórica mediante Monte Carlo")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "validacion_varianza.png", dpi=180, bbox_inches="tight")
plt.close()

# Figure 3: repeated means.
plt.figure(figsize=(7.4, 4.6))
plt.errorbar(
    biased["delta"],
    biased["mean_estimate"],
    yerr=biased["between_run_sd"],
    fmt="o",
    capsize=4,
    label="Media entre réplicas ± desviación entre réplicas",
)
plt.axhline(T_EXACT, linestyle="--", label="Transmisión exacta")
plt.xlabel(r"Factor de dilución, $\delta$")
plt.ylabel("Estimación de la transmisión")
plt.title("Estimaciones repetidas frente a la solución exacta")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "estimaciones_repetidas.png", dpi=180, bbox_inches="tight")
plt.close()

# Figure 4: relative efficiency benchmark.
plt.figure(figsize=(7.4, 4.6))
plt.plot(
    biased["delta"],
    biased["relative_efficiency_theory"],
    marker="o",
    label="Eficiencia con varianza teórica",
)
plt.plot(
    biased["delta"],
    biased["relative_efficiency_sample"],
    marker="s",
    label="Eficiencia con varianza muestral",
)
plt.axhline(1.0, linestyle="--", label="Referencia: Monte Carlo análogo")
plt.xlabel(r"Factor de dilución, $\delta$")
plt.ylabel("Eficiencia relativa")
plt.title("Ganancia computacional en la ejecución de referencia")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "eficiencia_relativa.png", dpi=180, bbox_inches="tight")
plt.close()
