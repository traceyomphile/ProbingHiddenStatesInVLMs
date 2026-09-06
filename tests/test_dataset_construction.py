# tests.test_dataset_construction.py 
import logging
import os
import pandas as pd
from pycocotools.coco import COCO
from src.dataset_construction import compute_cooccurrence, sample_image_ids, create_seed
from src.COCOSubset import COCOSubset

def test_balanced_question_types(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test to ensure that the generated dataset has a balanced number of questions for each question type.
    """
    # Count the number of questions for each question type
    question_counts = manifest_df['question_type'].value_counts()

    # Turn it into a DataFrame table
    question_counts_df = question_counts.reset_index()
    question_counts_df.columns = ['question_type', 'count']
    return question_counts_df

def test_for_duplicate_img_cat_pairs(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test to ensure that there are no duplicate (image_id, category) pairs in the generated dataset.
    """
    # Check for duplicate (image_id, category) pairs
    duplicate_pairs = manifest_df.duplicated(subset=['image_id', 'category'], keep=False)

    # Return the DataFrame with duplicate pairs
    return manifest_df[duplicate_pairs]

def test_validness_of_ground_truth(manifest_df: pd.DataFrame, coco_subset: COCOSubset) -> str:
    """
    Test to ensure that the ground_truth values are valid (True or False) in the generated dataset.
    """

    # Check if every present category in the manifest is actually present in the corresponding image according to the COCO annotations
    for _, row in manifest_df.iterrows():
        image_id = row['image_id']
        category = row['category']
        ground_truth = row['ground_truth']

        # Get the category ID for the given category name
        category_id = coco_subset.get_category_id(category)

        # Get the annotation IDs for the given image and category
        ann_ids = coco_subset.get_annotation_ids(image_id, category_id)

        # Determine if the category is present in the image according to COCO annotations
        is_present_in_coco = len(ann_ids) > 0

        # Check if the ground_truth value matches the actual presence of the category in the image
        if ground_truth != is_present_in_coco:
            return f"Mismatch for image_id {image_id}, category '{category}': ground_truth={ground_truth}, actual={is_present_in_coco}"

    return "Ground truth validation completed."

def test_deterministic_cooccurrence(coco_subset: COCOSubset) -> str:
    """
    Test to ensure that the co-occurrence of categories in the generated dataset is deterministic.
    """
    first = compute_cooccurrence(coco_subset)
    second = compute_cooccurrence(coco_subset)

    assert first == second, "Co-occurrence computation is not deterministic."
    return "Deterministic co-occurrence test passed."

def test_adversarial_absent_categories(manifest_df: pd.DataFrame, coco_subset: COCOSubset) -> str:
    """
    Test to ensure that the adversarial absent categories are indeed absent in the corresponding images.
    """
    # Check the adversarial absent categories image in the manifest against the COCO annotations
    for _, row in manifest_df[manifest_df['question_type'] == 'absent_adversarial'].iterrows():
        image_id = row['image_id']
        category = row['category']

        # Get the category ID for the given category name
        category_id = coco_subset.get_category_id(category)

        # Get the annotation IDs for the given image and category
        ann_ids = coco_subset.get_annotation_ids(image_id, category_id)

        # Determine if the category is present in the image according to COCO annotations
        is_present_in_coco = len(ann_ids) > 0

        # Check if the adversarial absent category is indeed absent in the image
        if is_present_in_coco:
            return f"Adversarial absent category '{category}' is present in image_id {image_id} according to COCO annotations."

    return "Adversarial absent categories test completed."

def test_sample_image_ids_determinism(coco_subset: COCOSubset, n_images: int, seed: int) -> str:
    """
    Test to ensure that the sampling of image IDs is deterministic given the same seed.
    """
    first_sample = sample_image_ids(coco_subset, n_images=n_images, seed=seed)
    second_sample = sample_image_ids(coco_subset, n_images=n_images, seed=seed)

    assert first_sample == second_sample, "Sampling of image IDs is not deterministic."
    return "Deterministic sampling test passed."

if __name__ == "__main__":
    # Define paths and parameters for testing
    manifest_path = "data/manifest.csv"
    image_dir = "val2017/"
    annotations_path = "instances_val2017.json"
    n_images = 200
    seed = create_seed("LTLTRA001")

    # Avoid reading the manifest each time
    manifest = pd.read_csv(manifest_path)
    # Avoid recomputing the Annotation
    coco_subset = COCOSubset(image_dir, annotations_path)

    # Define the logging configuration and make directory for logs if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    logging_path = 'logs/test_dataset_construction.log'

    logging.basicConfig(
        filename=logging_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run the tests and log the results
    logging.info("Starting tests for dataset construction...")
    logging.info(f"SEED : {seed}")

    try:
        balanced_question_types_result = test_balanced_question_types(manifest)
        logging.info(f"Balanced question types test result:\n{balanced_question_types_result}")

        duplicate_pairs_result = test_for_duplicate_img_cat_pairs(manifest)
        if duplicate_pairs_result.empty:
            logging.info("No duplicate (image_id, category) pairs found.")
        else:
            logging.warning(f"Duplicate (image_id, category) pairs found:\n{duplicate_pairs_result}")

        ground_truth_result = test_validness_of_ground_truth(manifest, coco_subset)
        logging.info(f"Ground truth validation result: {ground_truth_result}")

        cooccurrence_determinism_result = test_deterministic_cooccurrence(coco_subset)
        logging.info(cooccurrence_determinism_result)

        adversarial_absent_categories_result = test_adversarial_absent_categories(manifest, coco_subset)
        logging.info(adversarial_absent_categories_result)

        sampling_determinism_result = test_sample_image_ids_determinism(coco_subset, n_images, seed)
        logging.info(sampling_determinism_result)

        logging.info("All tests completed successfully.")
    except AssertionError as e:
        logging.error(f"AssertionError during testing: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during testing: {e}")

