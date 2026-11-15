# GeneralizationTest - Spectral seen-vs-unseen generalization-risk proxy
# Part of the Kapildhinwa2010-dev Toolkit.

"""Compare weight spectra between seen (training) and unseen (held-out)
layer matrices as a proxy for generalization risk.

Lineage: Reverse.Dean (b2/b4). This is a SPECTRAL PROXY only: it measures
how much the effective rank structure changes between seen and unseen
matrices. It does NOT measure the actual generalization gap (that requires
loss evaluations on real held-out data). Treat "most_unstable" as a layer
to investigate, not as proof of overfitting.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "layer_spectrum",
    "effective_rank",
    "layer_report",
    "analyze_model",
    "find_bottlenecks",
    "spectrum_gap",
    "generalization_report",
]


def layer_spectrum(weights: np.ndarray) -> np.ndarray:
    """Singular values of ``weights`` in descending order."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-D matrix")
    return np.linalg.svd(weights, compute_uv=False)


def effective_rank(weights: np.ndarray, energy: float = 0.99) -> int:
    """Smallest k whose singular values cover ``energy`` of total energy."""
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    s = layer_spectrum(weights)
    total = float(s.sum())
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(s) / total
    k = int(np.searchsorted(cumulative, energy, side="left") + 1)
    return int(min(k, s.size))


def layer_report(weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("weights must be a non-empty 2-D matrix")
    s = layer_spectrum(weights)
    s_max = float(s[0])
    s_min = float(s[-1])
    return {
        "shape": (int(weights.shape[0]), int(weights.shape[1])),
        "frobenius_norm": float(np.linalg.norm(weights)),
        "spectral_norm": s_max,
        # Numerically singular: s_min below machine-eps scale => inf
        "condition_number": (
            s_max / s_min
            if s_min > s_max * np.finfo(np.float64).eps * max(weights.shape)
            else float("inf")
        ),
        "effective_rank": effective_rank(weights),
    }


def analyze_model(layers: dict) -> dict:
    """Report every layer of a model dict {name: 2-D matrix}."""
    if not layers:
        raise ValueError("layers must be non-empty")
    return {name: layer_report(w) for name, w in layers.items()}


def find_bottlenecks(reports: dict, top_k: int = 1) -> list:
    """Layer names with the smallest effective rank, worst first."""
    if not reports:
        return []
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ranked = sorted(reports.items(), key=lambda kv: (kv[1]["effective_rank"], kv[0]))
    return [name for name, _ in ranked[:top_k]]


def spectrum_gap(seen: np.ndarray, unseen: np.ndarray,
                 energies=(0.5, 0.9, 0.99)) -> dict:
    """How much the rank structure changes from seen to unseen.

    Compares effective ranks at several energy levels and the spectral-norm
    ratio. A large positive rank gap means the unseen matrix uses a much
    different (usually smaller) subspace than the seen one.
    """
    if not all(0.0 < e <= 1.0 for e in energies):
        raise ValueError("all energies must be in (0, 1]")
    s_seen = layer_spectrum(seen)
    s_unseen = layer_spectrum(unseen)
    total_seen = float(s_seen.sum())
    total_unseen = float(s_unseen.sum())
    rank_gaps = {}
    for energy in energies:
        if total_seen <= 0.0:
            r_seen = 1
        else:
            cum = np.cumsum(s_seen) / total_seen
            r_seen = int(np.searchsorted(cum, energy, side="left") + 1)
        if total_unseen <= 0.0:
            r_unseen = 1
        else:
            cum = np.cumsum(s_unseen) / total_unseen
            r_unseen = int(np.searchsorted(cum, energy, side="left") + 1)
        rank_gaps[str(energy)] = int(r_seen - r_unseen)
    norm_seen = float(np.linalg.norm(seen))
    norm_unseen = float(np.linalg.norm(unseen))
    spectral_norm_ratio = (norm_unseen / norm_seen) if norm_seen > 0 else float("inf")
    return {
        "rank_gaps": rank_gaps,
        "spectral_norm_ratio": spectral_norm_ratio,
        "total_gap": int(sum(rank_gaps.values())),
    }


def generalization_report(seen_layers: dict, unseen_layers: dict) -> dict:
    """Per-layer seen-vs-unseen spectral gaps, worst layer first.

    Both dicts must have the same layer names mapping to 2-D matrices of
    compatible shape. Returns {"layers": {...}, "most_unstable": name}.
    """
    if not seen_layers or not unseen_layers:
        raise ValueError("both layer dicts must be non-empty")
    if set(seen_layers) != set(unseen_layers):
        raise ValueError("seen and unseen must contain the same layer names")
    per_layer = {}
    for name in seen_layers:
        per_layer[name] = spectrum_gap(seen_layers[name], unseen_layers[name])
    most_unstable = max(per_layer, key=lambda n: per_layer[n]["total_gap"])
    return {"layers": per_layer, "most_unstable": most_unstable}