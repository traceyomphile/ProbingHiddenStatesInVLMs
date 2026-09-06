# src.COCOSubset
import torch
import os
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCOSubset(Dataset):
    def __init__(self, img_dir: str | Path, coco_obj: COCO, category_names: list[str] | None = None, transform: transforms.Compose | None = None):
        self.img_dir = img_dir
        self.coco = coco_obj
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Get the image ID for the current idx.
        image_id = self.image_ids[idx]

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