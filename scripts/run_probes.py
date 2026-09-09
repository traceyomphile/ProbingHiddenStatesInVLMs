# srcipts.run_probes.py
import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.dataset_construction import create_seed
from src.inference import load_results, InferenceResult
from src.probing import _parse_image_id, train_val_split, build_feature_matrix, train_probe, evaluate_probe

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
    image_id = results[0].image_id

    assert len(results) == 1, (
        f'Expected exactly InferenceResult per checkpoint file,' 
        f'got {len(results)} for {path}'
    )

    n_layers = len(results[0].hidden_states)
    per_layer = {}
    for layer in range(n_layers):
        X, y = build_feature_matrix(results, layer, strategy)
        per_layer[layer] = (X, y)

    
    row_image_ids = np.array([image_id], dtype=np.int64)

    return per_layer, row_image_ids, n_layers

def get_features_streaming(checkpoints_paths: list[str], strategy: str, n_workers: int = 4):
    """
    Parallel pass over checkpoints, capped at `n_workers` files' worth of raw results in
    flight at once.
    """
    X_chunks: dict[int, list[np.ndarray]] = {}
    y_chunks: dict[int, list[np.ndarray]] = {}
    id_chunks: list[np.ndarray] = []
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
                    per_layer, row_image_ids, file_n_layers = future.result()
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
                progress.update(1)

    # Concatenate per-layer chunks into single arrays
    row_image_ids = np.concatenate(id_chunks, axis=0)

    features = {}
    for layer in range(n_layers):
        features[layer] = {
            'X': np.concatenate(X_chunks[layer], axis=0),
            'y': np.concatenate(y_chunks[layer], axis=0)
        }
        del X_chunks[layer], y_chunks[layer]        # Free chunk lists as we go

    return features, row_image_ids

def _build_row_masks(row_image_ids: np.ndarray, train_image_ids: np.ndarray, val_image_ids: np.ndarray):
    """
    Map per-row image_ids to boolean masks selecting train/val rows.
    Same masks apply to every later since row order is identical across layers
    """
    train_mask = np.isin(row_image_ids, train_image_ids)
    val_mask = np.isin(row_image_ids, val_image_ids)
    return train_mask, val_mask

if __name__ == '__main__':
    split_path = 'data/train_val_split.json'
    output_path = 'data/layer_auroc.csv'
    checkpoints_dir = Path('checkpoints')
    checkpoints_paths = [str(path) for path in checkpoints_dir.iterdir()]

    seed = create_seed('LTLTRA001')
    val_fraction = 0.3
    strategy = 'mean'

    # Split once and save
    train_image_ids, val_image_ids = get_or_create_split(Path(split_path), _get_image_ids(checkpoints_paths), val_fraction, seed)
    features, row_image_ids = get_features_streaming(checkpoints_paths, strategy)
    train_mask, val_mask = _build_row_masks(row_image_ids, train_image_ids, val_image_ids)

    # Define logger
    os.makedirs('logs', exist_ok=True)
    logging_path = 'logs/probing.log'

    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
    
