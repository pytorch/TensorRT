import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

SAMPLE_COUNT=5000
VAL_COUNT = 100
NUM_CLASSES = 150

class SceneParsingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, training=True): 
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        imgs = os.listdir(image_dir)
        if training:
            random.shuffle(imgs)
            # Picking a subset of the training dataset.
            # ADE20K Dataset has ~20K training samples.
            self.images = imgs[:SAMPLE_COUNT]
        else:
            self.images = imgs[:VAL_COUNT]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        img = None
        msk = None
        if not self.transform is None:
            transformed_data = self.transform(image=image, mask=mask)
            img = transformed_data["image"]
            msk = transformed_data["mask"]

        return img, msk
