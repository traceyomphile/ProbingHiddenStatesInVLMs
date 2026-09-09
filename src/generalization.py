# src/generalization.py

import numpy as np 
import pandas as pd 
from src.inference import InferenceResult
from src.probing import RowMeta

def _image_ids_to_indices(results: list[InferenceResult | RowMeta], image_ids: list[int]) -> list[int]:
    """
    Bridge between the two 'held-out set" representations used across the assignment.
    """
    wanted = set(int(i) for i in image_ids)
    return [i for i, result in enumerate(results) if result.image_id in wanted]

def cross_category_split(results: list[InferenceResult | RowMeta], train_types: list[str], test_types: list[str], val_indices: list[int]) -> tuple[list[int], list[int]]:
    """Builds a train/test split by question_type membership, restricted to indices
    already held out by Checkpoint C's train_val_split - this experiment must never
    train on an example that was in the within-category validation set."""
    train_types_set = set(train_types)
    test_types_set = set(test_types)

    overlap = train_types_set & test_types_set
    if overlap:
        raise ValueError(f'Train_types and test_types must be disjoint, gor overlap: {sorted(overlap)}')

    n_results = len(results)
    train_indices: list[int] = []
    test_indices: list[int] = []

    val_idx_set = {int(idx) for idx in val_indices}
    for idx in val_idx_set:
        if not (0 <= idx < n_results):
            raise IndexError(f'val_indices contains {idx}, out of range for {n_results} result')

    for idx, result in enumerate(results):
        question_type = _field(result, 'question_type')

        if idx not in val_idx_set and question_type in train_types_set:
            train_indices.append(idx)
        elif idx in val_idx_set and question_type in test_types_set:
            test_indices.append(idx)

        # else: question_type wasn't requested for this experiment - excluded from both

    return train_indices, test_indices

def align_metadata_by_row_id(metadata: list[RowMeta], row_ids: np.ndarray) -> list[RowMeta]:
    by_row_id: dict[int, RowMeta] = {}
    for row in metadata:
        if row.row_id in by_row_id:
            raise ValueError(f'Duplicate row_id {row.row_id} in metadata - row_id should be unique per checkpoint')
        by_row_id[row.row_id] = row

    try:
        return [by_row_id[int(row_id)] for row_id in row_ids]
    except KeyError as exc:
        raise KeyError(f'row_id {exc.args[0]} appears in row_ids but not in metadata') from exc

def _field(meta, name: str):
    """Read `name` off a metadata row that may be a plain dict or a dataclass"""
    if isinstance(meta, dict):
        return meta.get(name)
    return getattr(meta, name, None)

def compare_probe_to_baseline(probe_predictions: np.ndarray, confidence_predictions: np.ndarray, y_true: np.ndarray, metadata: list[dict]) -> pd.DataFrame:
    """
    Returns one row per validation example where the probe and the confidence
    baseline disagree, with enough metadata (image_id, category, question_type,
    ground_truth, confidence, probe_prediction) to read off Q9's examples directly '
    you should not be hand-assembing this table from printouts.
    """
    probe_predictions = np.asarray(probe_predictions)
    confidence_predictions = np.asarray(confidence_predictions)
    y_true = np.asarray(y_true)

    n = len(y_true)
    if not (len(probe_predictions) == len(confidence_predictions) == len(metadata) == n):
        raise ValueError(
            f'Length mismatch: probe_predictions={len(probe_predictions)}, '
            f'confidence_predictions={len(confidence_predictions)}, '
            f'y_true={n}, metadata={len(metadata)}'
        )

    disagreement_mask = probe_predictions != confidence_predictions

    rows = []
    for i in np.flatnonzero(disagreement_mask):
        meta = metadata[i]
        probe_pred = int(probe_predictions[i])
        baseline_pred = int(confidence_predictions[i])
        label = int(y_true[i])

        rows.append({
            'row_id': _field(meta, 'row_id'),
            'image_id': _field(meta, 'image_id'),
            'category': _field(meta, 'category'),
            'question_type': _field(meta, 'question_type'),
            'question': _field(meta, 'question'),
            'ground_truth': _field(meta, 'ground_truth'),
            'confidence': _field(meta, 'confidence'),
            'probe_prediction': probe_pred,
            'baseline_prediction': baseline_pred,
            'y_true': label,
            'winner': 'probe' if probe_pred == label else 'baseline',
        })

    columns = [
        'row_id', 'image_id', 'category', 'question_type', 'question', 'ground_truth', 'confidence',
        'probe_prediction', 'baseline_prediction', 'y_true', 'winner',
    ]

    return pd.DataFrame(rows, columns=columns)