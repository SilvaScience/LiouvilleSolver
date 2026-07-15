"""Momentum-space integration helpers."""

import warnings

import numpy as np


def resolve_k_weights(N_k, k_array=None, k_weights=None):
    """Return validated per-k integration weights."""
    if N_k is None:
        raise RuntimeError("Call feed_model() before resolving k weights.")

    if k_array is not None:
        k_array = np.asarray(k_array, dtype=float)
        if k_array.ndim != 1 or len(k_array) != N_k:
            raise ValueError(
                "k_array must be one-dimensional with length "
                f"{N_k}; got shape {k_array.shape}"
            )
        if not np.all(np.isfinite(k_array)):
            raise ValueError("k_array must contain only finite values")

    if k_weights is not None:
        raw_weights = np.asarray(k_weights)
        if np.iscomplexobj(raw_weights):
            raise ValueError("k_weights must be real")
        weights = np.asarray(raw_weights, dtype=float)
        if weights.ndim != 1 or len(weights) != N_k:
            raise ValueError(
                "k_weights must be one-dimensional with length "
                f"{N_k}; got shape {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("k_weights must contain only finite values")
        return weights.copy()

    if k_array is None or N_k == 1:
        return np.ones(N_k, dtype=float)

    warnings.warn(
        "Inferring rectangular integration weights from k_array uses the "
        "legacy endpoint convention. Pass explicit k_weights for periodic, "
        "trapezoidal, nonuniform, or multidimensional quadrature.",
        FutureWarning,
        stacklevel=3,
    )
    legacy_weight = (k_array[-1] - k_array[0]) / N_k / (2 * np.pi)
    return np.full(N_k, legacy_weight, dtype=float)


def integrate_k_response(response, k_weights, N_k, axis=-1):
    """Integrate one response array along its momentum axis."""
    response = np.asarray(response)
    axis = int(axis)
    if axis < 0:
        axis += response.ndim
    if axis < 0 or axis >= response.ndim:
        raise ValueError(f"axis {axis} is invalid for a {response.ndim}D response")
    if response.shape[axis] != N_k:
        raise ValueError(
            f"response momentum axis has length {response.shape[axis]}, "
            f"expected {N_k}"
        )
    shape = [1] * response.ndim
    shape[axis] = N_k
    return np.sum(response * np.asarray(k_weights).reshape(shape), axis=axis)
