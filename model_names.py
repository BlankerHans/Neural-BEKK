"""Canonical model identifiers and publication labels.

Single source of truth for model naming across every plot and table generator
(``evaluate_runs.py``, ``scripts/make_fhs_var_allmodels.py``,
``scripts/make_section4_loss_figures.py``, ...).

The keys mirror the model identifiers used in ``run_models.ipynb`` -- see the
``all_backtest_results`` dict in the "Save all Backtests" cell. Those keys are
the ground truth; anything rendered for the thesis must derive its label from
the mapping below rather than hard-coding its own.

Naming note: the ``bekk_lstm_*`` family are the MGARCH-LSTM / BEKK-kernel
hybrids (Zhao et al. 2024), labelled "BEKK-LSTM". The diagonal and full
``neural_bekk_asym_*`` specifications are the genuine "Neural-BEKK" models.
"""
from __future__ import annotations

# Canonical model order, following the notebook's narrative.
MODEL_ORDER = [
    "lstm_normal",
    "lstm_student",
    "gru_normal",
    "gru_student",
    "bekk_lstm_scalar",
    "bekk_lstm_vector",
    "bekk_lstm_mix",
    "neural_bekk_asym_diag",
    "neural_bekk_asym_full",
    "bekk_symmetric",
    "bekk_asymmetric",
    "dcc",
    "adcc",
]

# Publication-ready labels for tables and figures.
MODEL_LABELS = {
    "lstm_normal": "LSTM (Normal)",
    "lstm_student": "LSTM (Student-t)",
    "gru_normal": "GRU (Normal)",
    "gru_student": "GRU (Student-t)",
    "bekk_lstm_scalar": "BEKK-LSTM (scalar)",
    "bekk_lstm_vector": "BEKK-LSTM (vector)",
    "bekk_lstm_mix": "BEKK-LSTM (mix)",
    "neural_bekk_asym_diag": "Neural-BEKK (asym. diag.)",
    "neural_bekk_asym_full": "Neural-BEKK (asym. full)",
    "bekk_symmetric": "BEKK (symmetric)",
    "bekk_asymmetric": "BEKK (asymmetric)",
    "dcc": "DCC",
    "adcc": "ADCC",
}

# Legacy on-disk directory names -> canonical keys. Earlier seed runs saved the
# BEKK-LSTM hybrids under these names before they were renamed to ``bekk_lstm_*``;
# the stale duplicate directories still exist for some seeds.
LEGACY_ALIASES = {
    "neural_bekk_scalar": "bekk_lstm_scalar",
    "neural_bekk_vector": "bekk_lstm_vector",
    "bekk_mix": "bekk_lstm_mix",
}


def canonical_key(name: str) -> str:
    """Map a raw model identifier (possibly a legacy alias) to its canonical key."""
    return LEGACY_ALIASES.get(name, name)


def is_legacy_alias(name: str) -> bool:
    """True if ``name`` is a stale duplicate directory name, not a canonical key."""
    return name in LEGACY_ALIASES


def model_label(name: str) -> str:
    """Publication label for a model id; falls back to a humanised key."""
    key = canonical_key(name)
    return MODEL_LABELS.get(key, key.replace("_", " "))
