# srcipts.run_probes.py
import os
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.dataset_construction import create_seed
from src.inference import load_results
from src.probing import (
    _parse_image_id, 
    _parse_row_id,
    train_val_split, 
    build_feature_matrix, 
    train_probe, 
    evaluate_probe,
    evaluate_baseline,
    check_split_balance,
    save_metadata,
    correctness_label,
    _is_kept_row,
    RowMeta
)

def _get_image_ids(checkpoints: list[str]) -> list[int]:
    if checkpoints is None:
        raise ValueError('Checkpoints is none.')

    return [_parse_image_id(path) for path in checkpoints]

def get_or_create_split(split_path: Path, image_ids: list[int], val_frac: float, seed: int):
    """
    Load train/val split from disk if exists, otherwise, compute oce
    """
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)

        train_image_ids = np.array(split['train_image_ids'])
        val_image_ids = np.array(split['val_image_ids'])

        return train_image_ids, val_image_ids

    # Not cached yet: compute and persist.
    train_image_ids, val_image_ids = train_val_split(image_ids, val_frac, seed)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, 'w') as f:
        json.dump({
            'train_image_ids': np.asarray(train_image_ids).tolist(),
            'val_image_ids': np.asarray(val_image_ids).tolist(),
        }, f, indent=2)

    return train_image_ids, val_image_ids

def _load_and_extract(path: str, strategy: str):
    """
    Worker function: load one checkpoint's raw results.
    """
    results = load_results(path)

    assert len(results) == 1, (
        f'Expected exactly InferenceResult per checkpoint file,' 
        f'got {len(results)} for {path}'
    )

    row_id = _parse_row_id(path)
    result = results[0]

    n_layers = len(result.hidden_states)
    per_layer = {}
    for layer in range(n_layers):
        X, y = build_feature_matrix(results, layer, strategy)
        per_layer[layer] = (X, y)

    if _is_kept_row(result):
        row_image_ids = np.array([result.image_id], dtype=np.int32)  
        row_confidence = np.array([result.confidence], dtype=np.float64)
        row_label = np.array([correctness_label(result)], dtype=np.int16)
        row_ids = np.array([row_id], dtype=np.int16)
    else:
        row_image_ids = np.empty((0,), dtype=np.int32)      
        row_confidence = np.array((0,), dtype=np.float64)
        row_label = np.array((0,), dtype=np.int16)
        row_ids = np.array((0,), dtype=np.int16)

    # Get RowMeta for checking split balance
    row_meta = RowMeta(
        row_id=row_id,
        image_id=result.image_id,
        category=result.category,
        question_type=result.question_type,
        parsed_answer=result.parsed_answer,
        ground_truth=result.ground_truth,
        confidence=result.confidence,
    )

    return per_layer, row_image_ids, n_layers, row_confidence, row_label, row_ids, row_meta

def get_features_streaming(checkpoints_paths: list[str], strategy: str, n_workers: int = 4):
    """
    Parallel pass over checkpoints, capped at `n_workers` files' worth of raw results in
    flight at once.
    """
    X_chunks: dict[int, list[np.ndarray]] = {}
    y_chunks: dict[int, list[np.ndarray]] = {}
    id_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    row_id_chunks: list[np.ndarray] = []
    metadata: list[RowMeta] = []
    n_layers = None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_load_and_extract, path, strategy): path
            for path in checkpoints_paths
        }

        with tqdm(total=len(checkpoints_paths), desc='Loading + extracting', unit='file') as progress:
            for future in as_completed(futures):
                path = futures[future]

                try:
                    per_layer, row_image_ids, file_n_layers, row_confidence, row_label, row_ids, row_meta = future.result()
                except Exception:
                    logging.exception(f'Failed to load/extract featuress for path {path}')
                    raise

                if n_layers is None:
                    n_layers = file_n_layers
                    for layer in range(n_layers):
                        X_chunks[layer] = []
                        y_chunks[layer] = []

                for layer in range(n_layers):
                    X, y = per_layer[layer]
                    X_chunks[layer].append(X)
                    y_chunks[layer].append(y)

                id_chunks.append(row_image_ids)
                confidence_chunks.append(row_confidence)
                label_chunks.append(row_label)
                row_id_chunks.append(row_ids)
                metadata.append(row_meta)
                progress.update(1)

    # Concatenate per-layer chunks into single arrays
    row_image_ids = np.concatenate(id_chunks, axis=0)
    confidence = np.concatenate(confidence_chunks, axis=0)
    correctness = np.concatenate(label_chunks, axis=0)
    row_ids = np.concatenate(row_id_chunks, axis=0)

    features = {}
    for layer in range(n_layers):
        features[layer] = {
            'X': np.concatenate(X_chunks[layer], axis=0),
            'y': np.concatenate(y_chunks[layer], axis=0)
        }
        del X_chunks[layer], y_chunks[layer]        # Free chunk lists as we go

    return features, row_image_ids, confidence, correctness, row_ids, metadata

def _build_row_masks(row_image_ids: np.ndarray, train_image_ids: np.ndarray, val_image_ids: np.ndarray):
    """
    Map per-row image_ids to boolean masks selecting train/val rows.
    Same masks apply to every later since row order is identical across layers
    """
    train_mask = np.isin(row_image_ids, train_image_ids)
    val_mask = np.isin(row_image_ids, val_image_ids)
    return train_mask, val_mask

def _get_baseline(confidence: np.ndarray, y_true: np.ndarray) -> dict:
    return evaluate_baseline(confidence, y_true)
    
if __name__ == '__main__':
    split_path = 'data/train_val_split.json'
    output_path = 'data/layer_auroc.csv'
    baseline_output_path = 'data/baseline_comparison.csv'
    balance_output_path = 'data/split_balance.csv'
    metadata_output_path = 'data/row_metadata.json'
    checkpoints_dir = Path('checkpoints')
    checkpoints_paths = [str(path) for path in checkpoints_dir.iterdir()]

    seed = create_seed('LTLTRA001')
    val_fraction = 0.3
    strategy = 'mean'

    # Split once and save
    train_image_ids, val_image_ids = get_or_create_split(Path(split_path), _get_image_ids(checkpoints_paths), val_fraction, seed)
    features, row_image_ids, confidence, correctness, row_ids, metadata = get_features_streaming(checkpoints_paths, strategy)
    train_mask, val_mask = _build_row_masks(row_image_ids, train_image_ids, val_image_ids)

    # Persist metadata
    save_metadata(metadata, metadata_output_path)

    # Define logger
    os.makedirs('logs', exist_ok=True)
    logging_path = 'logs/probing.log'

    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Slit-balance sanity check:
    logging.info('Sanity Check-----')
    balance_df = check_split_balance(metadata, train_image_ids, val_image_ids)
    logging.info(f'Split balance:\n{balance_df}')

    Path(balance_output_path).parent.mkdir(parents=True, exist_ok=True)
    balance_df.to_csv(balance_output_path, index=False)

    # Run the tests and log the results
    logging.info("Starting probing...")

    results_rows = []
    for layer, data in features.items():
        X_train, y_train = data['X'][train_mask], data['y'][train_mask]
        X_val, y_val = data['X'][val_mask], data['y'][val_mask]

        probe = train_probe(X_train, y_train)
        eval_res = evaluate_probe(probe, X_val, y_val)

        eval_res['layer'] = layer
        results_rows.append(eval_res)

    
    eval_df = pd.DataFrame(results_rows, columns=['layer', 'accuracy', 'auroc'])
    logging.info(f'Probing results:\n{eval_df}')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)

    # --- Q7: best-layer probe vs condifence-only baseline baseline
    best_row = eval_df.loc[eval_df['auroc'].idxmax()]
    best_layer = int(best_row['layer'])

    baseline_res = _get_baseline(confidence[val_mask], correctness[val_mask])

    comparison_df = pd.DataFrame([
        {
            'method': f'Best-layer probe (layer {best_layer})',
            'accuracy': best_row['accuracy'],
            'auroc': best_row['auroc'],
        },
        {
            'method': 'Confidence-only baseline',
            'accuracy': baseline_res['accuracy'],
            'auroc': baseline_res['auroc'],
        }
    ])
    comparison_df['auroc_delta_vs_baseline'] = comparison_df['auroc'] - baseline_res['auroc']

    logging.info(f'Q7 comparison:\n{comparison_df}')

    Path(baseline_output_path).parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(baseline_output_path, index=False)

    
