# scripts.run_generalization.py

import json 
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from src.dataset_construction import load_manifest, create_seed
from src.generalization import compare_probe_to_baseline, cross_category_split
from src.probing import train_probe
from scripts.run_probes import get_features_streaming

TRAIN_TYPES = ('present', 'absent_random')
TEST_TYPES = ('absent_adversarial',)

def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return NaN when AUROC is undefined because only one class is present"""
    return float(roc_auc_score(y_true, scores) if np.unique(y_true).size == 2 else float('nan'))

def _load_split(path: Path) -> tuple[set[int], set[int]]:
    with path.open() as file:
        split = json.load(file)

    train_ids = {int(value) for value in split['train_image_ids']}
    val_ids = {int(value) for value in split['val_image_ids']}

    overlap = train_ids & val_ids
    if overlap:
        raise AssertionError(f'Part C train/validation image overlap: {sorted(overlap)[:10]}')

    return train_ids, val_ids

def _best_part_c_result(path: Path) -> pd.Series:
    results = pd.read_csv(path)
    required = {'layer', 'accuracy', 'auroc'}

    missing = required - set(results.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')

    valid = results.dropna(subset=['auroc'])
    if valid.empty:
        raise ValueError(f'{path} contains no defined AUROC value')

    return valid.loc[valid['auroc'].idxmax()]

def _manifest_rows(path: Path) -> list[dict]:
    rows = load_manifest(str(path))
    return rows.to_dict('records') if isinstance(rows, pd.DataFrame) else list(rows)

def _kept_metadata(metadata, row_ids: np.ndarray, manifest: list[dict]) -> list[dict]:
    """Align asynchronous checkpoint output with feature rows via row_id"""
    by_row_id = {int(row.row_id): row for row in metadata}
    if len(by_row_id) != len(metadata):
        raise ValueError('Duplicate row_id found in checkpoint metadata.')

    aligned = []
    for raw_row_id in row_ids:
        row_id = int(raw_row_id)
        if row_id not in by_row_id:
            raise KeyError(f'Feature row_id {row_id} has no checkpoint metadata')

        if not 0 <= row_id < len(manifest):
            raise IndexError(f'row_id {row_id} has no matching manifest row')

        meta = by_row_id[row_id]
        manifest_row = manifest[row_id]
        manifest_image_id = int(manifest_row['image_id'])
        if manifest_image_id != int(meta.image_id):
            raise ValueError(
                f'Manifest/checkpoint mismatch at row {row_id}: '
                f'{manifest_image_id} != {meta.image_id}'
            )

        aligned.append({
            'row_id': row_id,
            'image_id': int(meta.image_id),
            'category': meta.category,
            'question_type': meta.question_type,
            'question': str(manifest_row['question']),
            'ground_truth': bool(meta.ground_truth),
            'confidence': float(meta.confidence),
        })

    return aligned

def _write_audit(output_path: Path, metadata: list[dict], part_c_train_ids: set[int], part_c_val_ids: set[int], d_train_indices: list[int], d_test_indices: list[int]) -> pd.DataFrame:
    c_train_rows = {row['row_id'] for row in metadata if row['image_id'] in part_c_train_ids}
    c_val_rows = {row['row_id'] for row in metadata if row['image_id'] in part_c_val_ids}
    d_train_rows = {metadata[index]['row_id'] for index in d_train_indices}
    d_test_rows = {metadata[index]['row_id'] for index in d_test_indices}

    audit = pd.DataFrame([
        {'check': 'Part D train vs Part C held-out', 'overlap_count': len(d_train_rows  & c_val_rows)},
        {'check': 'Part D test vs Part C train', 'overlap_count': len(d_test_rows & c_train_rows)},
        {'check': 'Part D test vs Part D train', 'overlap_count': len(d_test_rows & d_train_rows)},
    ])
    audit.to_csv(output_path, index=False)

    failures = audit.loc[audit['overlap_count'] != 0]
    if not failures.empty:
        raise AssertionError(f'Checkpoint D disjointness failed:\n{failures.to_string(index=False)}')

    return audit

if __name__ == '__main__':
    checkpoints_dir = Path('checkpoints')
    manifest_path = Path('data/manifest.csv')
    split_path = Path('data/train_val_split.json')
    part_c_results_path = Path('data/layer_auroc.csv')
    output_dir = Path('data')

    strategy = 'mean'
    confidence_threshold = 0.65
    n_workers = 4
    seed = create_seed('LTLTRA001')

    output_dir.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    logging_path = 'logs/generalization.log'
    
    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    checkpoints_paths = sorted(str(path) for path in checkpoints_dir.glob('row_*.safetensors'))
    if not checkpoints_paths:
        raise FileNotFoundError(f'No row_*.safetensors files found in {checkpoints_dir}')

    part_c_train_ids, part_c_val_ids = _load_split(split_path)
    best_c = _best_part_c_result(part_c_results_path)
    best_layer = int(best_c['layer'])

    features, row_image_ids, confidence, correctness, row_ids, raw_metadata = get_features_streaming(checkpoints_paths, strategy, n_workers)
    if best_layer not in features:
        raise KeyError(f'Part C best layer {best_layer} is absent from extracted features')

    manifest = _manifest_rows(manifest_path)
    metadata = _kept_metadata(raw_metadata, row_ids, manifest)
    if not (len(metadata) == len(row_image_ids) == len(confidence) == len(correctness)):
        raise AssertionError('Feature arrays and aligned metadata have different lengths')

    # Convert Part C's held-out image IDs to row_indices before calling the supplied helper
    held_out_indices = [index for index, row in enumerate(metadata) if row['image_id'] in part_c_val_ids]
    d_train_indices, d_test_indices = cross_category_split(metadata, list(TRAIN_TYPES), list(TEST_TYPES), held_out_indices)
    if not d_train_indices or not d_test_indices:
        raise ValueError(f'Empty Part D split: {len(d_train_indices)} train rows, {len(d_test_indices)} test rows')

    audit = _write_audit(
        output_dir / 'checkpoint_d_disjointness.csv',
        metadata,
        part_c_train_ids,
        part_c_val_ids,
        d_train_indices,
        d_test_indices,
    )

    layer_data = features[best_layer]
    X, y = layer_data['X'], layer_data['y']

    train_idx = np.asarray(d_train_indices, dtype=int)
    test_idx = np.asarray(d_test_indices, dtype=int)
    if np.unique(y[train_idx]).size < 2:
        raise ValueError('Part D training labels contain only one class; logistic regression cannot be fitted')

    probe = train_probe(X[train_idx], y[train_idx])
    cross_predictions = probe.predict(X[test_idx]).astype(int)
    cross_scores = probe.predict_proba(X[test_idx])[:, 1]
    cross_eval = {
        'accuracy': float(accuracy_score(y[test_idx], cross_predictions)),
        'auroc': _safe_auroc(y[test_idx], cross_scores)
    }

    # Q8: compare the saved Part C witnin distribution result with cross-category transfer
    q8 = pd.DataFrame([
        {
            'setting': 'Within-distribution (Part C train -> validation)',
            'layer': best_layer,
            'n_train': int(np.isin(row_image_ids, list(part_c_train_ids)).sum()),
            'n_test': int(np.isin(row_image_ids, list(part_c_val_ids)).sum()),
            'accuracy': float(best_c['accuracy']),
            'auroc': float(best_c['auroc']),
        },
        {
            'setting': 'Cross-category (present/random -> adversarial)',
            'layer': best_layer,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'accuracy': float(cross_eval['accuracy']),
            'auroc': float(cross_eval['auroc']),
        },        
    ])

    q8['accuracy_gap_vs_within'] = float(best_c['accuracy']) - q8['accuracy']
    q8['auroc_gap_vs_within'] = float(best_c['auroc']) - q8['auroc']
    q8.to_csv(output_dir / 'generalization_gap.csv', index=False)

    probe_predictions = cross_predictions
    baseline_predictions = (confidence[test_idx] >= confidence_threshold).astype(int)
    test_metadata = [metadata[index] for index in d_test_indices]

    disagreements = compare_probe_to_baseline(probe_predictions, baseline_predictions, y[test_idx], test_metadata)
    disagreements = disagreements.sort_values('row_id', kind='stable').reset_index(drop=True)
    disagreements.to_csv(output_dir / 'disagreements.csv', index=False)

    # Deterministic
    q9 = disagreements.sample(n=min(3, len(disagreements)), random_state=(seed % (2**32)))
    q9['probe_prediction'] = q9['probe_prediction'].map({1: 'correct', 0: 'hallucinated'})
    q9['baseline_prediction'] = q9['baseline_prediction'].map({1: 'correct', 0: 'hallucinated'})
    q9.to_csv(output_dir / 'q9_examples.csv', index=False)

    baseline_metrics = pd.DataFrame([{
        'threshold': confidence_threshold,
        'n_test': len(test_idx),
        'accuracy': float(accuracy_score(y[test_idx], baseline_predictions)),
        'auroc': _safe_auroc(y[test_idx], confidence[test_idx]),
    }])
    baseline_metrics.to_csv(output_dir / 'q9_baseline_metrics.csv', index=False)

    logging.info(f'Checkpoint D audit:\n%s\n', audit.to_string(index=False))
    logging.info(f'Q8 generationalisation gap:\n%s\n', q8.to_string(index=False))
    logging.info(f'Q8 disagreements: %d\n', len(disagreements))
    logging.info(f'Q9 Disagreement examples:\n%s\n', q9.to_string(index=False))

    logging.info('Checkpoint D disjointness: PASS')
    logging.info(f'Best Part C layer {best_layer}')
    logging.info(f'Part D rows {len(train_idx)} train, {len(test_idx)} test')
    logging.info(f'Disagreements found: {len(disagreements)}')
    logging.info(f'Tables saved to: {output_dir}')