"""Monte Carlo transport model for a one-dimensional homogeneous slab.

The model assumes delta scattering: scattering events do not change the
particle direction.  Transmission is therefore controlled by absorption,
which provides a closed-form reference solution useful for validating the
Monte Carlo estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import process_time

import numpy as np


@dataclass(frozen=True)
class ShieldingModel:
    """Physical parameters of the homogeneous 1D slab."""

    thickness: float
    sigma_t: float
    sigma_s: float
    sigma_a: float

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")
        if min(self.sigma_t, self.sigma_s, self.sigma_a) < 0:
            raise ValueError("cross sections must be non-negative")
        if self.sigma_t <= 0:
            raise ValueError("sigma_t must be positive")
        if not np.isclose(self.sigma_t, self.sigma_s + self.sigma_a):
            raise ValueError("sigma_t must satisfy sigma_t = sigma_s + sigma_a")


@dataclass
class SimulationResult:
    """Tallies and timing information from one independent Monte Carlo run."""

    tallies: np.ndarray
    elapsed_cpu: float
    algorithm: str
    delta: float | None = None

    @property
    def n_histories(self) -> int:
        return int(self.tallies.size)

    @property
    def estimate(self) -> float:
        return float(np.mean(self.tallies))

    @property
    def tally_variance(self) -> float:
        if self.n_histories < 2:
            return float("nan")
        return float(np.var(self.tallies, ddof=1))

    @property
    def standard_error(self) -> float:
        return float(np.sqrt(self.tally_variance / self.n_histories))

    @property
    def cpu_time_per_history(self) -> float:
        return float(self.elapsed_cpu / self.n_histories)

    @property
    def efficiency(self) -> float:
        variance = self.tally_variance
        if variance <= 0 or self.elapsed_cpu <= 0:
            return float("inf")
        return float(1.0 / (self.cpu_time_per_history * variance))


def analytical_transmission(model: ShieldingModel) -> float:
    """Exact transmission for delta scattering."""

    return float(np.exp(-model.sigma_a * model.thickness))


def sample_free_path(rng: np.random.Generator, sigma: float) -> float:
    """Sample an exponential free path with macroscopic cross section sigma."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(rng.exponential(scale=1.0 / sigma))


def biased_second_moment(model: ShieldingModel, delta: float) -> float:
    """Exact E_b[Q^2] for the dilution-bias estimator.

    Under biased sampling all cross sections are multiplied by delta while
    collision-type probabilities are left unchanged.  The transmitted-history
    likelihood-ratio weight is

        Q = delta**(-n_s) * exp((delta - 1) * sigma_t * L),

    where n_s is the number of scattering events before escape.
    """

    _validate_delta(delta)
    exponent = model.thickness * (
        (delta - 2.0) * model.sigma_t + model.sigma_s / delta
    )
    return float(np.exp(exponent))


def biased_tally_variance(model: ShieldingModel, delta: float) -> float:
    """Exact variance of one biased-history tally."""

    transmission = analytical_transmission(model)
    return float(biased_second_moment(model, delta) - transmission**2)


def analog_tally_variance(model: ShieldingModel) -> float:
    """Exact Bernoulli variance of one analog-history tally."""

    transmission = analytical_transmission(model)
    return float(transmission * (1.0 - transmission))


def optimal_delta(model: ShieldingModel) -> float:
    """Variance-minimizing dilution factor for 0 < delta <= 1.

    For sigma_s > 0, differentiating the exact second moment gives
    delta* = sqrt(sigma_s / sigma_t).  For pure absorption, the infimum is
    approached as delta -> 0+, so this function returns 0.0 to represent that
    boundary case analytically; simulations still require delta > 0.
    """

    if model.sigma_s == 0:
        return 0.0
    return float(np.sqrt(model.sigma_s / model.sigma_t))


def simulate_analog(
    model: ShieldingModel,
    n_histories: int,
    seed: int | None = None,
) -> SimulationResult:
    """Run the analog Monte Carlo transport estimator in vectorized batches.

    Histories are advanced simultaneously while they remain active. This keeps
    the event-by-event transport logic but avoids a Python loop over every
    particle history.
    """

    _validate_n_histories(n_histories)
    rng = np.random.default_rng(seed)
    p_abs = model.sigma_a / model.sigma_t

    tallies = np.zeros(n_histories, dtype=float)
    positions = np.zeros(n_histories, dtype=float)
    active = np.ones(n_histories, dtype=bool)

    start = process_time()
    while np.any(active):
        idx = np.flatnonzero(active)
        free_paths = rng.exponential(scale=1.0 / model.sigma_t, size=idx.size)
        new_positions = positions[idx] + free_paths

        transmitted = new_positions >= model.thickness
        transmitted_idx = idx[transmitted]
        tallies[transmitted_idx] = 1.0
        active[transmitted_idx] = False

        collision_idx = idx[~transmitted]
        if collision_idx.size == 0:
            continue

        absorbed = rng.random(collision_idx.size) < p_abs
        absorbed_idx = collision_idx[absorbed]
        active[absorbed_idx] = False

        scattered_idx = collision_idx[~absorbed]
        positions[scattered_idx] = new_positions[~transmitted][~absorbed]

    elapsed_cpu = process_time() - start

    return SimulationResult(
        tallies=tallies,
        elapsed_cpu=elapsed_cpu,
        algorithm="analog",
        delta=None,
    )


def simulate_biased(
    model: ShieldingModel,
    n_histories: int,
    delta: float,
    seed: int | None = None,
) -> SimulationResult:
    """Run dilution biasing with exact likelihood-ratio correction.

    Active histories are advanced in vectorized batches. At ``delta=1`` this
    routine follows the same random-number draws and collision decisions as
    :func:`simulate_analog`, enabling a history-by-history identity test.
    """

    _validate_n_histories(n_histories)
    _validate_delta(delta)

    rng = np.random.default_rng(seed)
    sigma_t_biased = delta * model.sigma_t
    p_abs = model.sigma_a / model.sigma_t
    attenuation_weight = np.exp(
        (sigma_t_biased - model.sigma_t) * model.thickness
    )

    tallies = np.zeros(n_histories, dtype=float)
    positions = np.zeros(n_histories, dtype=float)
    n_scatter = np.zeros(n_histories, dtype=np.int64)
    active = np.ones(n_histories, dtype=bool)

    start = process_time()
    while np.any(active):
        idx = np.flatnonzero(active)
        free_paths = rng.exponential(scale=1.0 / sigma_t_biased, size=idx.size)
        new_positions = positions[idx] + free_paths

        transmitted = new_positions >= model.thickness
        transmitted_idx = idx[transmitted]
        if transmitted_idx.size:
            tallies[transmitted_idx] = (
                delta ** (-n_scatter[transmitted_idx])
            ) * attenuation_weight
            active[transmitted_idx] = False

        collision_idx = idx[~transmitted]
        if collision_idx.size == 0:
            continue

        absorbed = rng.random(collision_idx.size) < p_abs
        absorbed_idx = collision_idx[absorbed]
        active[absorbed_idx] = False

        scattered_idx = collision_idx[~absorbed]
        if scattered_idx.size:
            positions[scattered_idx] = new_positions[~transmitted][~absorbed]
            n_scatter[scattered_idx] += 1

    elapsed_cpu = process_time() - start
    return SimulationResult(
        tallies=tallies,
        elapsed_cpu=elapsed_cpu,
        algorithm="biased",
        delta=float(delta),
    )


def _validate_delta(delta: float) -> None:
    if not (0.0 < delta <= 1.0):
        raise ValueError("delta must satisfy 0 < delta <= 1")


def _validate_n_histories(n_histories: int) -> None:
    if not isinstance(n_histories, (int, np.integer)) or n_histories <= 1:
        raise ValueError("n_histories must be an integer greater than 1")
