# scripts.build_dataset.py
from pathlib import Path
import src.dataset_construction as dataset_construction
# Must produce data/manifest.csv

if __name__ == "__main__":
    try:
        # Create a seed
        STUDENT_NUMBER = "LTLTRA001"
        seed = dataset_construction.create_seed(STUDENT_NUMBER)

        # Define the annotation file path
        annotation_path = '../instances_val2017.json'
        image_dir = '../val2017'

        # Create the COCOSubset object
        coco_subset = dataset_construction.load_coco_subset(annotation_path, image_dir)

        # Compute the category co-occurences
        co_occurrences = dataset_construction.compute_cooccurrence(coco_subset)

        # Sample 200 images
        sampled_image_ids = dataset_construction.sample_image_ids(coco_subset, n_images=200, seed=seed)

        # Build the question set
        question_set = dataset_construction.build_question_set(coco_subset, sampled_image_ids, co_occurrences, seed=seed)

        # Save the question set to a CSV file
        output_csv_path = "../data/manifest.csv"
        dataset_construction.save_manifest(question_set, output_csv_path)

        print(f"Dataset construction completed successfully. Manifest saved to {output_csv_path}")
    except Exception as e:
        print(f"An error occurred: {e}")