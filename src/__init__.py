"""COMPAS fairness analysis package."""

from .data_loader import DEFAULT_DATASET_PATH, load_dataset
from .fairness_metrics import compute_confusion_counts, compute_group_metrics, compute_race_metrics
from .preprocessing import apply_propublica_filters

__all__ = [
    "DEFAULT_DATASET_PATH",
    "load_dataset",
    "apply_propublica_filters",
    "compute_confusion_counts",
    "compute_group_metrics",
    "compute_race_metrics",
]
