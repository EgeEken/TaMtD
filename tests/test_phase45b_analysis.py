import numpy as np
import pytest

from tamtd.analysis.phase45b import (
    fit_linear_model,
    paired_correctness,
    pareto_efficient,
    prediction_metrics,
    validate_manifest_alignment,
)


def test_prediction_metrics_uses_train_class_prior_and_reports_bits():
    logits = np.array([[4.0, 0.0], [0.0, 4.0]])
    labels = np.array([0, 1])
    result = prediction_metrics(logits, labels, np.array([0, 0, 1]))
    assert result["accuracy"] == 1.0
    assert result["cross_entropy_nats"] < 0.03
    assert result["class_prior_cross_entropy_nats"] > result["cross_entropy_nats"]
    assert result["u_probe_bits"] > 0


def test_paired_correctness_counts_and_bootstrap_are_deterministic():
    labels = np.array([0, 0, 1, 1, 2])
    a = np.array([0, 1, 1, 0, 2])
    b = np.array([0, 0, 0, 1, 2])
    result = paired_correctness(labels, a, b, bootstrap_seed=17, bootstrap_replicates=500)
    again = paired_correctness(labels, a, b, bootstrap_seed=17, bootstrap_replicates=500)
    assert (result["both_correct"], result["a_only_correct"], result["b_only_correct"], result["both_wrong"]) == (2, 1, 2, 0)
    assert result["bootstrap_ci95_low"] == again["bootstrap_ci95_low"]
    assert result["bootstrap_ci95_high"] == again["bootstrap_ci95_high"]


def test_pareto_marks_strict_dominance_only():
    mask = pareto_efficient(np.array([0.0, 1.0, 2.0, 1.0]), np.array([0.2, 0.5, 0.4, 0.5]))
    assert mask.tolist() == [True, True, False, True]


def test_fit_linear_model_has_loo_predictions_and_codec_effects():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = fit_linear_model({"x": x}, 1.0 + 2.0 * x, np.array(["a", "a", "b", "b", "c"]))
    assert result["n"] == 5
    assert len(result["loo_predictions"]) == 5
    assert "codec[b]" in result["coefficient_names"]
    assert result["r_squared"] == pytest.approx(1.0)


def test_manifest_alignment_checks_identity_and_hash():
    expected = [{"cache_index": 0, "image_id": "x", "label": 1, "split": "test", "rgb_sha256": "abc"}]
    validate_manifest_alignment(expected, [dict(expected[0])])
    changed = [dict(expected[0], image_id="y")]
    with pytest.raises(ValueError, match="alignment"):
        validate_manifest_alignment(expected, changed)
