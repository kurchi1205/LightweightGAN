from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision
from torchvision import transforms
from functools import lru_cache, partial
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


def load_data(data_name, sample=None):
    # "julianmoraes/doodles-captions-BLIP"
    dataset = load_dataset(data_name, split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(sample))
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) * 0.1)

    train = dataset.select(range(train_size))
    test = dataset.select(range(train_size, train_size + test_size))
    val = dataset.select(range(train_size + test_size, len(dataset)))
    return train, test, val


class ImageDataset(Dataset):
    def __init__(
        self,
        data,
        image_size,
        transparent = False,
        greyscale = False,
        aug_prob = 0.5,
    ):
        super().__init__()
        self.data = data
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(p=0.5, max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]["image"]
        img = np.array(img, dtype=np.float32)
        img = (img - 127.5)/255
        return self.transform(image=img)
    

def get_data(data_name, image_size, aug_prob, sample=None):
    train, test, val = load_data(data_name, sample)

    if aug_prob is None and len(train) < 1e5:
        aug_prob = min(0.5, (1e5 - len(train)) * 3e-6)
        print(f'autosetting augmentation probability to {round(aug_prob * 100)}%')

    train_dataset = ImageDataset(train, image_size=image_size, aug_prob=aug_prob)
    val_dataset = ImageDataset(val, image_size=image_size, aug_prob=aug_prob)
    test_dataset = ImageDataset(test, image_size=image_size, aug_prob=0)
    return train_dataset, val_dataset, test_dataset