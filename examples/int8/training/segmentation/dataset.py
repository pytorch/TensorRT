import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

SAMPLE_COUNT=20000
class SceneParsingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, training=True): 
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        imgs = os.listdir(image_dir)
        if training:
            random.shuffle(imgs)
            self.images = imgs[:SAMPLE_COUNT]
        else:
            self.images = imgs

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        img = None
        msk = None
        if not self.transform is None:
            transformed_data = self.transform(image=image, mask=mask)
            img = transformed_data["image"]
            msk = transformed_data["mask"]

        return img, msk