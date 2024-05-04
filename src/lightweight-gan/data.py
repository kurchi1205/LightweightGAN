from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision
from torchvision import transforms
from functools import lru_cache, partial
from PIL import Image

def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


def load_data():
    dataset = load_dataset("julianmoraes/doodles-captions-BLIP", split="train")
    dataset = dataset.shuffle(seed=42)

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
        aug_prob = 0.
    ):
        super().__init__()
        self.data = data
        dataset = dataset.shuffle(seed=42)

        num_channels = 3
        self.transform = transforms.Compose([
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=aug_prob),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.data.image[index]
        return self.transform(img)
    

def get_data(image_size, aug_prob):
    train, test, val = load_data()

    if aug_prob is None and len(train) < 1e5:
        aug_prob = min(0.5, (1e5 - len(train)) * 3e-6)
        print(f'autosetting augmentation probability to {round(aug_prob * 100)}%')

    train_dataset = ImageDataset(train, image_size=image_size, aug_prob=aug_prob)
    val_dataset = ImageDataset(val, image_size=image_size, aug_prob=aug_prob)
    test_dataset = ImageDataset(test, image_size=image_size, aug_prob=0)
    return train_dataset, val_dataset, test_dataset