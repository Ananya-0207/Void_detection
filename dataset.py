import os
import glob
import random
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.mask_utils import json_to_mask


#ImageNet statistics (required for pretrained MobileNetV2) 
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


#Augmentation pipelines 
#Strong augmentation pipeline for training.albumentations applies all spatial transforms to both image and mask jointly.
    
def get_train_transforms(h=config.INPUT_H, w=config.INPUT_W) -> A.Compose:
   
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
        ),
        A.ElasticTransform(
            alpha=30, sigma=5, p=0.3,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.6
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

"""Validation and test pipeline: resize + normalise only."""
def get_val_transforms(h=config.INPUT_H, w=config.INPUT_W) -> A.Compose:
    
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


#Helper: discover all matched image/JSON pairs in a folder

def find_pairs(dataset_dir: str,
               void_label: str = config.VOID_LABEL):
    image_exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    all_images = []

    # search inside dataset and all subfolders
    for ext in image_exts:
        all_images.extend(
            glob.glob(
                os.path.join(dataset_dir, "**", f"*{ext}"),
                recursive=True
            )
        )

    all_images = sorted(set(all_images))

    pairs = []
    missing_json = []

    for img_path in all_images:

        stem = Path(img_path).stem

        # json should be inside same folder as image
        json_path = os.path.join(
            os.path.dirname(img_path),
            stem + ".json"
        )

        if os.path.exists(json_path):
            pairs.append((img_path, json_path))
        else:
            missing_json.append(img_path)

    if missing_json:
        print(f"⚠ {len(missing_json)} image(s) have no matching JSON — skipped:")
        for p in missing_json[:5]:
            print(os.path.basename(p))

    if not pairs:
        raise FileNotFoundError(
            f"No image/JSON pairs found in: {dataset_dir}\n"
            f"Make sure images and .json files have same stem name."
        )

    print(f"Found {len(pairs)} matched image/JSON pairs")
    return sorted(pairs)

# Main Dataset class

class VoidDataset(Dataset):
    def __init__(self,
                 pairs:      List[Tuple[str, str]],
                 transform:  Optional[A.Compose] = None,
                 void_label: str = config.VOID_LABEL):
        self.pairs      = pairs
        self.transform  = transform
        self.void_label = void_label

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, json_path = self.pairs[idx]

        #Load image (RGB, uint8) 
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3]

        #  Parse JSON → binary mask (0/1) 
        mask = json_to_mask(json_path,
                            target_label=self.void_label)   # [H, W] uint8

        #  Apply transforms 
        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]   # torch.Tensor [3, H, W]
            mask  = out["mask"]    # torch.Tensor [H, W]
        else:
            image = torch.from_numpy(
                image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask)

        return image, mask.long()   # mask must be long for CrossEntropyLoss

    def get_raw_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the original un-transformed (image_rgb, mask) for visualisation.
        """
        img_path, json_path = self.pairs[idx]
        bgr   = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask  = json_to_mask(json_path, target_label=self.void_label)
        return image, mask

    def filename(self, idx: int) -> str:
        """Return the stem name of sample idx (e.g. 'img_000')."""
        return Path(self.pairs[idx][0]).stem


#  Split and build DataLoaders 

def make_splits(dataset_dir: str = config.DATASET_DIR,
                train_size:  int = config.TRAIN_SIZE,
                val_size:    int = config.VAL_SIZE,
                test_size:   int = config.TEST_SIZE,
                seed:        int = config.RANDOM_SEED
               ) -> Tuple[List, List, List]:
   
    pairs = find_pairs(dataset_dir)

    total = len(pairs)
    need  = train_size + val_size + test_size
    if total < need:
        raise ValueError(
            f"Only {total} matched pairs found, but split requires "
            f"{train_size}+{val_size}+{test_size}={need}.\n"
            f"Either add more images/JSONs, or reduce TRAIN/VAL/TEST sizes in config.py."
        )

    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

#converting ratio to counts
    total=len(shuffled)
    if isinstance(train_size, float):
        train_size = int(train_size * total)
    if isinstance(val_size, float):
        val_size = int(val_size * total)
    if isinstance(test_size, float):
        test_size = int(test_size * total)
    #final splits
    train = shuffled[:train_size]
    val   = shuffled[train_size:train_size+val_size]
    test  = shuffled[train_size + val_size : train_size + val_size + test_size]

    print(f"  Split → train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def get_dataloaders(dataset_dir: str = config.DATASET_DIR,
                    batch_size:  int  = config.BATCH_SIZE,
                    num_workers: int  = config.NUM_WORKERS) -> dict:
  
    train_pairs, val_pairs, test_pairs = make_splits(dataset_dir)

    pin = torch.cuda.is_available()

    return {
        "train": DataLoader(
            VoidDataset(train_pairs, transform=get_train_transforms()),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin, drop_last=True,
        ),
        "val": DataLoader(
            VoidDataset(val_pairs, transform=get_val_transforms()),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        ),
        "test": DataLoader(
            VoidDataset(test_pairs, transform=get_val_transforms()),
            batch_size=1, shuffle=False,   # one image at a time for per-image metrics
            num_workers=num_workers, pin_memory=pin,
        ),
        # keep raw dataset objects for visualisation
        "_train_ds": VoidDataset(train_pairs, transform=None),
        "_val_ds"  : VoidDataset(val_pairs,   transform=None),
        "_test_ds" : VoidDataset(test_pairs,  transform=None),
    }


#  Quick sanity check ─
if __name__ == "__main__":
    print("Testing dataset loader …\n")
    loaders = get_dataloaders(config.DATASET_DIR)

    for split in ("train", "val", "test"):
        ldr   = loaders[split]
        imgs, masks = next(iter(ldr))
        print(f"  {split:<5}  images={imgs.shape}  masks={masks.shape}  "
              f"unique_mask_vals={masks.unique().tolist()}")
