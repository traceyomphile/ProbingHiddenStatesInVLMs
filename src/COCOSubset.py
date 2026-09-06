# src.COCOSubset
import os
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCOSubset(Dataset):
    def __init__(self, img_dir: str | Path, annotation_file: str | Path):
        self.img_dir = img_dir
        self.annotation_file = annotation_file

        # Create COCO object and load the annotations
        self.coco = COCO(self.annotation_file)
        # Ensure that the image IDs are sorted for deterministic behavior
        self.img_ids = sorted(self.coco.getImgIds())

        # Build a mapping from category IDs to category names
        self.category_mapping = {cat["id"]: cat["name"] for cat in self.coco.loadCats(sorted(self.coco.getCatIds()))}
        # Build a mapping from image IDs to their present categories
        self.image_ids_to_categories = self._build_image_id_to_categories()
        

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 1. Get the image ID for the current idx.
        image_id = self.img_ids[idx]

        # 2. Load image metadata from COCO
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.img_dir, image_info['file_name'])

        # 3. Load the image using PIL
        image = Image.open(image_path)

        return image

    def get_image_ids(self) -> list[int]:
        return self.img_ids

    def get_category_names(self) -> list[str]:
        return list(self.category_mapping.values())

    def _build_image_id_to_categories(self) -> dict[int, list[str]]:
        image_id_to_categories = {}
        # Sort annotation ids for deterministic behavior
        annotation_ids = sorted(self.coco.getAnnIds())
        for ann in self.coco.loadAnns(annotation_ids):
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            category_name = self.category_mapping[category_id]

            if image_id not in image_id_to_categories:
                image_id_to_categories[image_id] = []

            # Check for duplicates before appending
            if category_name not in image_id_to_categories[image_id]:
                image_id_to_categories[image_id].append(category_name)

        # Sort the categories for deterministic behavior
        for image_id in image_id_to_categories:
            image_id_to_categories[image_id].sort()
        return image_id_to_categories

    def get_present_categories(self, image_id: int) -> list[str]:
        if image_id not in self.img_ids:
            raise ValueError(f"Image ID {image_id} not found in the dataset.")
        
        if self.image_ids_to_categories is None:
            raise ValueError("Could not build the mapping of image IDs to categories.")

        return self.image_ids_to_categories.get(image_id, [])

    def get_filename(self, image_id: int) -> str:
        if image_id not in self.img_ids:
            raise ValueError(f"Image ID {image_id} not found in the dataset.")

        image_info = self.coco.loadImgs(image_id)[0]
        return image_info['file_name']