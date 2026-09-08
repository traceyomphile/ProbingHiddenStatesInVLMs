# scripts.run_inference.py
import torch
import os
import logging
import numpy as np
import pandas as pd
from collections import Counter
from src.dataset_construction import load_manifest, create_seed
from src.inference import load_model, run_inference_resumable, load_checkpoint_metadata

def _get_unclear_rate(metadata_results: list[dict]) -> pd.DataFrame:
    strict_unclear_count, with_fallback_count, unresolveable = 0, 0, 0

    for item in metadata_results:
        used_fallback = item['parse_result']['used_fallback']
        parsed_answer = item['parsed_answer']

        if used_fallback:
            strict_unclear_count += 1

            if parsed_answer is None:
                unresolveable += 1
            else:
                with_fallback_count += 1

    total = len(metadata_results)
    
    results = {
        'strict_unclear_rate': (strict_unclear_count / total),
        'fallback_recovered_rate': (with_fallback_count / total),
        'truly_unresolved_rate': (unresolveable / total)
    }

    return pd.DataFrame(results.items(), columns=['rate_type', 'rate'])

def _get_accuracy(metadata_results: list[dict]) -> tuple[float, Counter]:
    total_correct = 0
    total_len = len(metadata_results)

    qt_correct = Counter()
    qt_total = Counter()

    for item in metadata_results:
        question_type = item['question_type']

        qt_total[question_type] += 1

        if item['parsed_answer'] is not None and (item['parsed_answer'] == item['ground_truth']):
            total_correct += 1
            qt_correct[question_type] += 1

    qt_acc = {
        question_type: qt_correct[question_type] / count
        for question_type, count in qt_total.items()
    }

    return (total_correct / total_len), qt_acc

if __name__ == '__main__':
    manifest_path = 'data/manifest.csv'
    checkpoint_dir = 'checkpoints'
    model_name = 'HuggingFaceTB/SmolVLM-256M-Instruct'
    image_dir = 'val2017'

    # Get device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Get seed
    seed = create_seed('LTLTRA001')
    rng = np.random.default_rng(seed)

    # Load manifest:
    manifest = load_manifest(manifest_path)
    # Load SmolVLM-256M-Instruct
    model, processor = load_model(model_name, device)

    # Run inference on full manifest
    checkpoint_paths = run_inference_resumable(model, processor, manifest, image_dir, checkpoint_dir)

    # Reload saved inference artefacts from disk
    metadata_results = load_checkpoint_metadata(checkpoint_paths)

    # Define the logging configuration and make directory for logs if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    logging_path = 'logs/inference.log'

    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run the tests and log the results
    logging.info("Starting inference checkpoints...")

    # Q3: Get unclear rate
    unclear_results = _get_unclear_rate(metadata_results)
    logging.info(f"Unclear model responses:\n{unclear_results}")

    # Q4
    overall_acc, qt_acc = _get_accuracy(metadata_results)
    qt_df = pd.DataFrame(qt_acc.items(), columns=['question_type', 'accuracy'])
    logging.info(f'Overall answer accuracy:\n{overall_acc}')
    logging.info(f'Accuracy per question_type:\n{qt_df}')

    
    

    
    
