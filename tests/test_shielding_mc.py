import numpy as np
import pytest

from src.shielding_mc import (
    ShieldingModel,
    analog_tally_variance,
    analytical_transmission,
    biased_tally_variance,
    optimal_delta,
    sample_free_path,
    simulate_analog,
    simulate_biased,
)


MODEL = ShieldingModel(
    thickness=100.0,
    sigma_t=0.1,
    sigma_s=0.05,
    sigma_a=0.05,
)


def test_cross_sections_must_be_consistent():
    with pytest.raises(ValueError):
        ShieldingModel(100.0, 0.1, 0.04, 0.05)


def test_analytical_transmission_matches_expected_expression():
    expected = np.exp(-MODEL.sigma_a * MODEL.thickness)
    assert analytical_transmission(MODEL) == pytest.approx(expected)


def test_free_path_sample_mean_is_consistent_with_exponential_law():
    rng = np.random.default_rng(2026)
    samples = np.array([sample_free_path(rng, MODEL.sigma_t) for _ in range(100_000)])
    assert samples.mean() == pytest.approx(1.0 / MODEL.sigma_t, rel=0.015)


def test_delta_one_reproduces_analog_history_by_history():
    analog = simulate_analog(MODEL, n_histories=10_000, seed=77)
    biased = simulate_biased(MODEL, n_histories=10_000, delta=1.0, seed=77)
    np.testing.assert_array_equal(biased.tallies, analog.tallies)


def test_biased_estimator_converges_to_analytical_solution():
    result = simulate_biased(MODEL, n_histories=200_000, delta=0.70, seed=91)
    exact = analytical_transmission(MODEL)
    assert abs(result.estimate - exact) <= 4.0 * result.standard_error


def test_exact_bias_variance_reduces_to_analog_variance_at_delta_one():
    assert biased_tally_variance(MODEL, 1.0) == pytest.approx(
        analog_tally_variance(MODEL)
    )


def test_analytical_optimum_is_local_minimum_of_exact_variance():
    delta_star = optimal_delta(MODEL)
    center = biased_tally_variance(MODEL, delta_star)
    left = biased_tally_variance(MODEL, delta_star - 0.02)
    right = biased_tally_variance(MODEL, delta_star + 0.02)
    assert center < left
    assert center < right
