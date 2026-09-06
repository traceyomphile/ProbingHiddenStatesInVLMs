# src.dataset_construction.py
import json
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from pathlib import Path
from pycocotools.coco import COCO

from src.COCOSubset import COCOSubset

# PART A 
def create_seed(student_number: str):
    """
    Create a seed from the given student number. The student number is used to seed the random number generator.
    Args:
        student_number (str): The student number to seed the random number generator.
    """
    seed = int.from_bytes(student_number.encode('utf-8'), byteorder='big')
    return seed

def load_coco_subset(annotation_path: str, image_dir: str) -> "COCOSubset":
    """
    Wraps pycocotools access to the provided COCO subset.
    """
    return COCOSubset(img_dir=image_dir, annotation_file=annotation_path)

def compute_cooccurrence(coco: COCOSubset) -> dict[tuple[str, str], int]:
    """
    Symmetric pairwise category co-occurrence counts across every image in coco.
    """
    # Get all category names
    category_names = coco.get_category_names()

    # Generate combinations of category pairs
    category_pairs = combinations(category_names, 2)

    # Initialize co-occurrence counter
    cooccurrence = Counter()

    # Count co-occurrences across all images
    for image_id in coco.get_image_ids():
        present_categories = coco.get_present_categories(image_id)
        for cat1, cat2 in category_pairs:
            if cat1 in present_categories and cat2 in present_categories:
                cooccurrence[(cat1, cat2)] += 1

    return dict(cooccurrence)

def sample_image_ids(coco: COCOSubset, n_images: int, seed: int) -> list[int]:
    """
    Deterministic image-ID sample for the given seed. Calling this twice with the
    same seed must return identical output - this is one of your sanity checks.
    """
    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed)
    all_image_ids = coco.get_image_ids()

    # Sample n_images unique image IDs without replacement
    sampled_ids = rng.choice(all_image_ids, size=n_images, replace=False)
    return sampled_ids.tolist()

def build_question_set(coco: COCOSubset, image_ids: list[int], cooccurrence: dict[tuple[str, str], int], seed: int) -> list[dict]:
    """
    Returns exactly 3 records per image ID: one each of question_type
    "present" / "absent_random" / "absent_adversarial", each a dict with keys
    image_id, category, question, question_type, ground_truth.
    """
    vowels = ['a', 'e', 'i', 'o', 'u']
    question_set = []
    for image_id in image_ids:
        present_categories = coco.get_present_categories(image_id)
        all_categories = coco.get_category_names()
        absent_categories = [cat for cat in all_categories if cat not in present_categories]

        # Randomly select one present category
        rng = np.random.default_rng(seed + image_id)  # Ensure different seed per image
        if present_categories:
            present_category = rng.choice(present_categories)

            # Check if present_category starts with a vowel for question phrasing
            if present_category[0].lower() in vowels:
                question_set.append({
                    "image_id": image_id,
                    "category": present_category,
                    "question": f"Is there an '{present_category}' in this image? Answer yes or no.",
                    "question_type": "present",
                    "ground_truth": True
                })
            else:
                question_set.append({
                    "image_id": image_id,
                    "category": present_category,
                    "question": f"Is there a '{present_category}' in this image? Answer yes or no.",
                    "question_type": "present",
                    "ground_truth": True
                })

        # Randomly select one absent category for absent_random
        if absent_categories:
            absent_random_category = rng.choice(absent_categories)

            # Check if absent_random_category starts with a vowel for question phrasing
            if absent_random_category[0].lower() in vowels:
                question_set.append({
                    "image_id": image_id,
                    "category": absent_random_category,
                    "question": f"Is there an '{absent_random_category}' in this image? Answer yes or no.",
                    "question_type": "absent_random",
                    "ground_truth": False
                })
            else:
                question_set.append({
                    "image_id": image_id,
                    "category": absent_random_category,
                    "question": f"Is there a '{absent_random_category}' in this image? Answer yes or no.",
                    "question_type": "absent_random",
                    "ground_truth": False
                })

        # For absent_adversarial, select an absent category that has high co-occurrence with a present category
        if present_categories and absent_categories:
            cooccurring_absent = [
                cat for cat in absent_categories 
                if any((present_cat, cat) in cooccurrence or (cat, present_cat) in cooccurrence for present_cat in present_categories)
            ]
            if cooccurring_absent:
                # Choose the absent adversarial category with the highest co-occurrence count
                absent_adversarial_category = max(cooccurring_absent, key=lambda cat: max(cooccurrence.get((present_cat, cat), 0) for present_cat in present_categories))

                # Check if absent_adversarial_category starts with a vowel for question phrasing
                if absent_adversarial_category[0].lower() in vowels:
                    question_set.append({
                        "image_id": image_id,
                        "category": absent_adversarial_category,
                        "question": f"Is there an '{absent_adversarial_category}' in this image? Answer yes or no.",
                        "question_type": "absent_adversarial",
                        "ground_truth": False
                    })
                else:
                    question_set.append({
                        "image_id": image_id,
                        "category": absent_adversarial_category,
                        "question": f"Is there a '{absent_adversarial_category}' in this image? Answer yes or no.",
                        "question_type": "absent_adversarial",
                        "ground_truth": False
                    })

    return question_set

    
def save_manifest(questions: list[dict], path: str | Path) -> None:
    """Saves the question set to a file at the specified path."""
    # Define acceptable file extensions
    acceptable_extensions = {'.json', '.jsonl', '.csv'}
    if not isinstance(path, Path):
        path = Path(path)
    if path.suffix not in acceptable_extensions:
        raise ValueError("Unsupported file extension. Please use .json, .jsonl, or .csv.")

    # If the file extension is .json, save as a JSON file
    if path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(questions, f, indent=4)

    # If the file extension is .jsonl, save as a JSON Lines file
    elif path.suffix == '.jsonl':
        with open(path, 'w') as f:
            for question in questions:
                f.write(json.dumps(question) + '\n')

    # If the file extension is .csv, save as a CSV file
    elif path.suffix == '.csv':
        df = pd.DataFrame(questions)
        df.to_csv(path, index=False)
    

def load_manifest(path: str | Path) -> list[dict]:
    """Loads the question set from a file at the specified path."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    # If the file extension is .json, load as a JSON file
    if path.suffix == '.json':
        with open(path, 'r') as f:
            questions = json.load(f)

    # If the file extension is .jsonl, load as a JSON Lines file
    elif path.suffix == '.jsonl':
        questions = []
        with open(path, 'r') as f:
            for line in f:
                questions.append(json.loads(line.strip()))

    # If the file extension is .csv, load as a CSV file
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
        questions = df.to_dict(orient='records')

    else:
        raise ValueError("Unsupported file extension. Please use .json, .jsonl, or .csv.")

    return questions