# src.dataset_construction.py
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter
from pathlib import Path
from src.COCOSubset import COCOSubset

# PART A 
def create_seed(student_number: str) -> int:
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
    # Initialize co-occurrence counter
    cooccurrence = Counter()

    # Count co-occurrences across all images
    for image_id in coco.get_image_ids():
        present_categories = coco.get_present_categories(image_id)
        combinations_of_categories = combinations(sorted(present_categories), 2)
        for pair in combinations_of_categories:
            cooccurrence[pair] += 1

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
    all_categories = coco.get_category_names()
    question_set = []
    rng = np.random.default_rng(seed)

    for image_id in image_ids:
        present_categories = coco.get_present_categories(image_id)
        absent_categories = [cat for cat in all_categories if cat not in present_categories]

        # Choose present cat and adversarial negative as the pair in cooccurrence with the highest count
        best_pair = None
        best_count = -1
        for (cat_a, cat_b), count in cooccurrence.items():
            if cat_a in present_categories and cat_b in absent_categories and count > best_count:
                best_pair = (cat_a, cat_b)
                best_count = count
            elif cat_a in absent_categories and cat_b in present_categories and count > best_count:
                best_pair = (cat_b, cat_a)
                best_count = count

        if best_pair is None:
            # Fallback: no valid adversarial pair found for this image
            present_category = rng.choice(present_categories)
            absent_adversarial_category = rng.choice(absent_categories)
        else:
            present_category, absent_adversarial_category = best_pair

        # Avoid removing elements
        random_absent_pool = [cat for cat in absent_categories if cat != absent_adversarial_category]
        absent_random_category = rng.choice(random_absent_pool)

        question_set.append({
            "image_id": image_id,
            "category": present_category,
            "question": f"Is there a {present_category} in this image? Answer yes or no.",
            "question_type": "present",
            "ground_truth": True
        })

        question_set.append({
            "image_id": image_id,
            "category": absent_random_category,
            "question": f"Is there a {absent_random_category} in this image? Answer yes or no.",
            "question_type": "absent_random",
            "ground_truth": False
        })

        question_set.append({
            "image_id": image_id,
            "category": absent_adversarial_category,
            "question": f"Is there a {absent_adversarial_category} in this image? Answer yes or no.",
            "question_type": "absent_adversarial",
            "ground_truth": False
        })

    return question_set

    
def save_manifest(questions: list[dict], path: str) -> None:
    """Saves the question set to a csv file at the specified path."""
    if not isinstance(path, Path):
        path = Path(path)

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save as a CSV file
    df = pd.DataFrame(questions)
    df.to_csv(path, index=False)
    

def load_manifest(path: str) -> list[dict]:
    """Loads the question set from a file at the specified path."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    # Load as a CSV file
    df = pd.read_csv(path)
    questions = df.to_dict(orient='records')

    return questions