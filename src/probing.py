# src/probing.py
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from src.inference import InferenceResult
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

# Actual probe implementation

PROBING_STRATEGIES = ['mean', 'last_token', 'max']
LABELS = {1: 'correct', 0: 'hallucinated'}
QUESTION_TYPES = ('present', 'absent_random', 'absent_adversarial')
FILENAME_RE = re.compile(r"row_(\d{6})_image_(\d{12})\.safetensors")

@dataclass(frozen=True)
class RowMeta:
    row_id: int
    image_id: int
    category: str
    question_type: str
    parsed_answer: bool | None
    ground_truth: bool | None 
    confidence: float
def _parse_row_id(path: str) -> int:
    """Extract the unique row number from a checkpoint file without loading the file."""
    match = FILENAME_RE.search(Path(path).name)
    if not match:
        raise ValueError(f'Could not parse image_id from filename: {path}')
    return int(match.group(1))

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

def _is_kept_row(result: InferenceResult) -> bool:
    return result.parsed_answer is not None

def correctness_label(result: InferenceResult) -> int:
    return 1 if result.parsed_answer == result.ground_truth else 0

def build_feature_matrix(results: list[InferenceResult], layer: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) for one layer across all examples with a non-unclear label."""
    X, y = [], []
    for result in results:
        if not _is_kept_row(result):
            continue

        X.append(pool_layer(result.hidden_states[layer], strategy))
        y.append(correctness_label(result))

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

def evaluate_baseline(confidence: np.ndarray, y_true: np.ndarray, threshold: float = 0.65) -> dict:
    """
    Baseline: predict correctness using only the model's own generation confidence.
    """
    predictions = (confidence >= threshold).astype(int)

    accuracy = accuracy_score(y_true, predictions)
    auroc = roc_auc_score(y_true, confidence)

    return {
        'accuracy': accuracy,
        'auroc': auroc
    }

def check_split_balance(results: list[InferenceResult | RowMeta], train_image_ids: np.ndarray, val_image_ids: np.ndarray, q_types: tuple[str] = QUESTION_TYPES) -> pd.DataFrame:
    """
    Sanity check the train/val split
    """
    train_ids = set(int(i) for i in train_image_ids)
    val_ids = set(int(i) for i in val_image_ids)

    def _summarize(rows: list, split_name: str, question_type: str) -> dict:
        unclear = [r for r in rows if not _is_kept_row(r)]
        kept = [r for r in rows if _is_kept_row(r)]
        n_correct = sum(correctness_label(r) == 1 for r in kept)
        n_hallucinated = len(kept) - n_correct

        return {
            'split': split_name,
            'question_type': question_type,
            'n_total': len(rows),
            'n_unclear': len(unclear),
            'n_correct': n_correct,
            'n_hallucinated': n_hallucinated,
            'pct_correct': (n_correct / len(kept) if kept else float('nan'))
        }

    report_rows = []
    for split_name, id_set in [('train', train_ids), ('val', val_ids)]:
        split_rows = [r for r in results if r.image_id in id_set]

        # Union so an unexpected question type in the data still shows up
        question_types = sorted(set(q_types) | {r.question_type for r in split_rows})

        for q_type in q_types:
            group = [r for r in split_rows if r.question_type == q_type]
            report_rows.append(_summarize(group, split_name, q_type))

        report_rows.append(_summarize(split_rows, split_name, 'ALL'))

    columns = [
        'split', 'question_type', 'n_total', 'n_unclear', 'n_correct', 
        'n_hallucinated', 'pct_correct',
    ]
    balanced_df = pd.DataFrame(report_rows, columns=columns)

    return balanced_df

def save_metadata(metadata: list[RowMeta], path: str | Path) -> None:
    """
    Persist RowMeta rows as JSON, so downstream work can reload them withouut re-reading every
    checkpoint's hidden states again. One JSON object per row, keyed by row_id
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump([asdict(row) for row in metadata], f, indent=2)

def load_metadata(path: str | Path) -> list[RowMeta]:
    """Load RowMeta rows previouslt written by save_metadata()."""
    with open(path) as f:
        raw_rows = json.load(f)
    return [RowMeta(**row) for row in raw_rows]

def row_meta_to_metadata_dicts(metadata: list[RowMeta]) -> list[dict]:
    return [
        {
            'image_id': row.image_id,
            'category': row.category,
            'question_type': row.question_type,
            'ground_truth': row.ground_truth,
            'confidence': row.confidence,
        }
        for row in metadata
    ]