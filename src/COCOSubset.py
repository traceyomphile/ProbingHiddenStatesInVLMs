# src.COCOSubset
import torch
import os
import json
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCOSubset(Dataset):
    def __init__(self, img_dir: str | Path, annotation_file: str | Path, category_names: list[str] | None = None, transform: transforms.Compose | None = None):
        self.img_dir = img_dir
        self.annotation_file = annotation_file
        self.category_names = category_names
        self.transform = transform

        # Load COCO annotation data
        with open(self.annotation_file, 'r') as f:
            self.data = json.load(f)
        self.coco = COCO(self.annotation_file)
        self.img_ids = self.coco.getImgIds()
        self.category_mapping = {cat["id"]: cat["name"] for cat in self.data["categories"]}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # 1. Get the image ID for the current idx.
        image_id = self.img_ids[idx]

        # 2. Load image metadata from COCO
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.img_dir, image_info['file_name'])

        # 3. Load the image using PIL and convert it to RGB.
        image = Image.open(image_path).convert('RGB')

        # Load the annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        if self.transform:
            image = self.transform(image)

        return image

    def get_image_ids(self) -> list[int]:
        return self.img_ids

    def get_category_names(self) -> list[str]:
        if self.data is None:
            raise ValueError("COCO data not loaded. Please ensure the annotation file is provided.")
    
        category_names = [cat["name"] for cat in self.data["categories"]]
        return category_names

    def get_category_ids(self) -> list[int]:
        if self.data is None:
            raise ValueError("COCO data not loaded. Please ensure the annotation file is provided.")
    
        category_ids = [cat["id"] for cat in self.data["categories"]]
        return category_ids

    def get_category_names(self, category_id: int) -> list[str] | None:
        if self.data is None:
            raise ValueError("COCO data not loaded. Please ensure the annotation file is provided.")
    
        category_name = self.category_mapping.get(category_id, None)
        return category_name

    def get_present_categories(self, image_id: int) -> list[str]:
        if self.data is None:
            raise ValueError("COCO data not loaded. Please ensure the annotation file is provided.")
    
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        present_category_ids = {ann["category_id"] for ann in annotations}
        present_categories = [self.category_mapping[cat_id] for cat_id in present_category_ids]
        return present_categories

    def get_filename(self, image_id: int) -> str:
        if self.data is None:
            raise ValueError("COCO data not loaded. Please ensure the annotation file is provided.")
    
        image_info = self.coco.loadImgs(image_id)[0]
        return image_info['file_name']