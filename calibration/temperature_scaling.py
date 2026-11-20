"""Global temperature scaling and probability-quality metrics for option logits.

All calculations use float64. Padding is represented by negative infinity and
has zero probability. ECE uses top-label confidence and right-closed equal-width
bins. Brier is the sum over classes, averaged over examples.
"""

import numpy as np


def _validate_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (nonempty examples, at least 2 options)")
    if np.any(np.isnan(logits)) or np.any(np.isposinf(logits)):
        raise ValueError("logits may contain finite values and negative-infinity padding only")
    if np.any(np.isfinite(logits).sum(axis=1) < 2):
        raise ValueError("each example must contain at least 2 finite option logits")
    return logits


def _validate_labels(logits, labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("labels must be a one-dimensional integer array aligned to logits")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer option indices")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("a label is outside the option-index range")
    if np.any(~np.isfinite(logits[np.arange(len(logits)), labels])):
        raise ValueError("labels must refer to finite, valid options")
    return labels.astype(np.int64, copy=False)


def _log_probabilities(logits, temperature):
    if np.ndim(temperature) != 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite scalar")
    with np.errstate(over="ignore", under="ignore"):
        centered = logits - np.max(logits, axis=1, keepdims=True)
        scaled = centered / float(temperature)
        normalizer = np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    return scaled - normalizer


def probabilities(logits, temperature=1.0):
    """Return valid-option softmax probabilities after dividing logits by T."""
    logits = _validate_logits(logits)
    return np.exp(_log_probabilities(logits, temperature))


def evaluate_logits(logits, labels, temperature=1.0, n_bins=15):
    """Compute accuracy, NLL, multiclass Brier, top-label ECE, and bin data.

    ECE bins are ((m-1)/M, m/M], with zero included in the first bin.
    Accuracy and ECE are fractions. Empty-bin statistics are None.
    """
    logits = _validate_logits(logits)
    labels = _validate_labels(logits, labels)
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    log_probabilities = _log_probabilities(logits, temperature)
    probs = np.exp(log_probabilities)
    rows = np.arange(len(logits))
    predictions = np.argmax(logits, axis=1)
    correct = predictions == labels
    confidence = probs[rows, predictions]
    squared_error = np.square(probs).sum(axis=1) - 2.0 * probs[rows, labels] + 1.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        selected = bin_indices == index
        count = int(selected.sum())
        bin_accuracy = float(correct[selected].mean()) if count else None
        bin_confidence = float(confidence[selected].mean()) if count else None
        gap = bin_confidence - bin_accuracy if count else None
        if count:
            ece += count * abs(gap) / len(logits)
        bins.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "accuracy": bin_accuracy,
            "confidence": bin_confidence,
            "gap": gap,
        })
    return {
        "n": len(logits),
        "accuracy": float(correct.mean()),
        "nll": float(-log_probabilities[rows, labels].mean()),
        "brier": float(squared_error.mean()),
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "bins": bins,
    }


def _nll_and_derivative(logits, labels, inverse_temperature):
    """Evaluate convex NLL(beta) and its derivative for validated inputs."""
    log_probs = _log_probabilities(logits, 1.0 / inverse_temperature)
    probs = np.exp(log_probs)
    centered = logits - np.max(logits, axis=1, keepdims=True)
    finite_values = np.where(np.isfinite(centered), centered, 0.0)
    expected = (probs * finite_values).sum(axis=1)
    true_values = centered[np.arange(len(logits)), labels]
    return float(-log_probs[np.arange(len(logits)), labels].mean()), float((expected - true_values).mean())


def fit_temperature(logits, labels):
    """Fit one positive T by minimizing example-mean calibration NLL.

    The derivative is monotone in beta=1/T. Bisection solves its zero within
    beta in [1e-4, 1e4]. A constrained optimum reports boundary='lower' or
    'upper'. Uninformative tied logits return T=1. No evaluation data enter
    this function unless explicitly supplied by its caller.
    """
    logits = _validate_logits(logits)
    labels = _validate_labels(logits, labels)
    nll_before, _ = _nll_and_derivative(logits, labels, 1.0)
    lower, upper = 1e-4, 1e4
    boundary = None
    iterations = 0
    converged = True
    finite = np.isfinite(logits)
    minimum = np.where(finite, logits, np.inf).min(axis=1)
    if np.all(minimum == logits.max(axis=1)):
        beta = 1.0
    else:
        _, derivative_lower = _nll_and_derivative(logits, labels, lower)
        _, derivative_upper = _nll_and_derivative(logits, labels, upper)
        if derivative_lower >= 0:
            beta = lower
            boundary = "lower"
        elif derivative_upper <= 0:
            beta = upper
            boundary = "upper"
        else:
            converged = False
            for iterations in range(1, 101):
                beta = (lower + upper) / 2.0
                _, derivative = _nll_and_derivative(logits, labels, beta)
                if derivative < 0:
                    lower = beta
                else:
                    upper = beta
                if upper - lower <= 1e-10 * max(1.0, abs(beta)):
                    converged = True
                    break
            beta = (lower + upper) / 2.0
    nll_after, _ = _nll_and_derivative(logits, labels, beta)
    if not np.isfinite(nll_before) or not np.isfinite(nll_after):
        raise ValueError("calibration NLL is nonfinite for the supplied logit range")
    if nll_after > nll_before + 1e-10:
        raise RuntimeError("temperature fitting increased calibration NLL")
    return {
        "temperature": float(1.0 / beta),
        "inverse_temperature": float(beta),
        "nll_before": nll_before,
        "nll_after": nll_after,
        "converged": converged,
        "boundary": boundary,
        "iterations": iterations,
        "n": len(logits),
    }
