# src/probing.py
import re
import numpy as np
from pathlib import Path
from src.inference import InferenceResult
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# Actual probe implementation

PROBING_STRATEGIES = ['mean', 'last_token', 'max']
LABELS = {1: 'correct', 0: 'hallucinated'}
FILENAME_RE = re.compile(r"row_(\d{6})_image_(\d{12})\.safetensors")

def _parse_image_id(path: str) -> int:
    """Extract image_id from a checkpoint filename without loading the file."""
    match = FILENAME_RE.search(Path(path).name)
    if not match:
        raise ValueError(f'Could not parse image_id from filename: {path}')
    return int(match.group(2))

def pool_layer(hidden_states: np.ndarray, strategy: str) -> np.ndarray:
    """
    (seq_len, hidden_dim) -> (hidden_dim,). strategy is one of the options you
    justified in Q5 - keep it a named parameter, not a hardcoded choice, so you can
    compare strategies later if you choose to.
    """
    if strategy.lower() == 'mean':
        return np.mean(hidden_states, axis=0)
    if strategy.lower() == 'max':
        return np.max(hidden_states, axis=0)
    if strategy.lower() == 'last_token':
        return hidden_states[-1]
    raise ValueError(f'Strategies allowed: {PROBING_STRATEGIES}')

def build_feature_matrix(results: list[InferenceResult], layer: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) for one layer across all examples with a non-unclear label."""
    X, y = [], []
    for result in results:
        parsed_answer = result.parsed_answer

        # Skip the non-clear labels
        if parsed_answer is None:
            continue

        label = 1 if parsed_answer == True else 0
        X.append(pool_layer(result.hidden_states[layer], strategy))
        y.append(label)

    return np.array(X), np.array(y)

def train_val_split(image_ids: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """
    Returns (train_indices, val_indices), stratified by question_type. Call this
    ONCE and reuse the same split for every layer and every probe in Parts C and D -
    a fresh random split per layer would silently leak information across your
    generalises across layers" comparisons in Q6.

    NOTE: Changed assignment doc signature as it is expensive to load the whole results.
    """
    unique_ids = np.unique(sorted(image_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)

    n_val = int(len(shuffled) * val_fraction)
    val_indices = sorted(shuffled[:n_val].tolist())
    train_indices = sorted(shuffled[n_val:].tolist())

    return train_indices, val_indices

def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    probe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000))
    ])
    probe.fit(X_train, y_train)

    return probe

def evaluate_probe(probe, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Returns at least {"accuracy": float, "auroc": float}."""
    predictions = probe.predict(X_val)
    pred_scores = probe.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predictions)
    auroc = roc_auc_score(y_val, pred_scores)

    return {
        'accuracy': accuracy,
        'auroc': auroc,
    }