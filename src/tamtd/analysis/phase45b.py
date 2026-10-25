from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def validate_manifest_alignment(expected: list[dict], actual: list[dict]) -> None:
    if len(expected) != len(actual):
        raise ValueError(f"manifest row counts differ: {len(expected)} != {len(actual)}")
    fields = ("cache_index", "image_id", "label", "split", "rgb_sha256")
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if any(left.get(key) != right.get(key) for key in fields):
            raise ValueError(f"manifest alignment differs at row {index}")


def prediction_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    train_labels: np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != len(labels) or len(labels) == 0:
        raise ValueError("logits must have shape [samples, classes] and align with labels")
    classes = logits.shape[1]
    if np.any(labels < 0) or np.any(labels >= classes) or np.any(train_labels < 0) or np.any(train_labels >= classes):
        raise ValueError("labels fall outside the logit class range")
    counts = np.bincount(train_labels, minlength=classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("every class must have training examples")
    prior = counts / counts.sum()
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probs)
    rows = np.arange(len(labels))
    true_log_probs = log_probs[rows, labels]
    predictions = logits.argmax(axis=1)
    one_hot = np.zeros_like(probabilities)
    one_hot[rows, labels] = 1.0
    model_ce = float(-true_log_probs.mean())
    prior_ce = float(-np.log(prior[labels]).mean())
    label_counts = np.bincount(labels, minlength=classes).astype(np.float64)
    test_prior = label_counts / label_counts.sum()
    entropy = float(-np.sum(test_prior[test_prior > 0] * np.log(test_prior[test_prior > 0])))
    return {
        "n": int(len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "cross_entropy_nats": model_ce,
        "class_prior_cross_entropy_nats": prior_ce,
        "test_label_entropy_nats": entropy,
        "u_probe_nats": prior_ce - model_ce,
        "u_probe_bits": (prior_ce - model_ce) / math.log(2.0),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "predicted_labels": predictions,
        "probabilities": probabilities,
        "per_image_cross_entropy_nats": -true_log_probs,
    }


def paired_correctness(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    bootstrap_seed: int = 20261025,
    bootstrap_replicates: int = 10000,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions_a = np.asarray(predictions_a, dtype=np.int64)
    predictions_b = np.asarray(predictions_b, dtype=np.int64)
    if labels.ndim != 1 or predictions_a.shape != labels.shape or predictions_b.shape != labels.shape or len(labels) == 0:
        raise ValueError("paired labels and predictions must be equal nonempty vectors")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    correct_a = predictions_a == labels
    correct_b = predictions_b == labels
    both = int(np.sum(correct_a & correct_b))
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))
    rng = np.random.default_rng(bootstrap_seed)
    probabilities = np.asarray([only_a, only_b, both + neither], dtype=np.float64) / len(labels)
    draws = rng.multinomial(len(labels), probabilities, size=bootstrap_replicates)
    differences = (draws[:, 0] - draws[:, 1]) / len(labels)
    discordant = only_a + only_b
    p_value = float(binomtest(min(only_a, only_b), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    union = int(np.sum(correct_a | correct_b))
    return {
        "n": int(len(labels)),
        "both_correct": both,
        "a_only_correct": only_a,
        "b_only_correct": only_b,
        "both_wrong": neither,
        "accuracy_a": float(correct_a.mean()),
        "accuracy_b": float(correct_b.mean()),
        "accuracy_difference_a_minus_b": float(correct_a.mean() - correct_b.mean()),
        "bootstrap_ci95_low": float(np.quantile(differences, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(differences, 0.975)),
        "mcnemar_exact_p": p_value,
        "prediction_agreement": float(np.mean(predictions_a == predictions_b)),
        "correctness_jaccard": float(both / union) if union else 1.0,
    }


def pareto_efficient(x: np.ndarray, utility: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    if x.ndim != 1 or utility.shape != x.shape:
        raise ValueError("x and utility must be aligned vectors")
    efficient = np.ones(len(x), dtype=bool)
    for index in range(len(x)):
        dominates = (x <= x[index]) & (utility >= utility[index]) & ((x < x[index]) | (utility > utility[index]))
        if np.any(dominates):
            efficient[index] = False
    return efficient


def fit_linear_model(
    predictors: dict[str, np.ndarray],
    response: np.ndarray,
    categories: np.ndarray | None = None,
) -> dict:
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 1 or len(response) < 3:
        raise ValueError("linear model needs at least three response values")
    columns = [np.ones(len(response))]
    names = ["intercept"]
    for name, values in predictors.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != response.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid predictor: {name}")
        columns.append(values)
        names.append(name)
    if categories is not None:
        category_values = np.asarray(categories, dtype=str)
        if category_values.shape != response.shape:
            raise ValueError("categories must align with response")
        levels = sorted(set(category_values.tolist()))
        for level in levels[1:]:
            columns.append((category_values == level).astype(np.float64))
            names.append(f"codec[{level}]")
    design = np.column_stack(columns)
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
    residuals = response - fitted
    total = float(np.sum((response - response.mean()) ** 2))
    residual_sum = float(np.sum(residuals**2))
    r_squared = 1.0 - residual_sum / total if total > 0 else 1.0
    n, parameters = design.shape
    adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / (n - parameters) if n > parameters else None
    loo = np.full(n, np.nan, dtype=np.float64)
    for index in range(n):
        keep = np.arange(n) != index
        coef, _, _, _ = np.linalg.lstsq(design[keep], response[keep], rcond=None)
        loo[index] = float(design[index] @ coef)
    return {
        "n": int(n),
        "parameters": int(parameters),
        "rank": int(rank),
        "coefficient_names": names,
        "coefficients": coefficients.tolist(),
        "r_squared": float(r_squared),
        "adjusted_r_squared": None if adjusted is None else float(adjusted),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "loo_rmse": float(np.sqrt(np.mean((response - loo) ** 2))),
        "fitted": fitted,
        "residuals": residuals,
        "loo_predictions": loo,
    }
