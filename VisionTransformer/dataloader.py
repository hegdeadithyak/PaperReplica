import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__all__ = ["CustomDataset"]


# Augment train data
train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

# Don't augment test data, only reshape
test_transforms = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any class in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class CustomDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(
            self.data_dir
        )  # find classes and class_to_idx

    def load_image(self, index: int) -> Image.Image:
        # This should be changed as per your dataset structure
        # image_path = os.path.join(self.data_dir,self.classes[index//100],f'{index%100}.jpg')
        return Image.open(image_path)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        image = self.load_image(index)
        label = self.class_to_idx[image_path.parent.name]

        if self.transform:
            image = self.transform(image)
            return image, label
        else:
            return image, label


# # Turn train and test custom Dataset's into DataLoader's
# from torch.utils.data import DataLoader

# train_dataloader_custom = DataLoader(
#     dataset=train_data_custom,  # use custom created train Dataset
#     batch_size=1,  # how many samples per batch?
#     num_workers=0,  # how many subprocesses to use for data loading? (higher = more)
#     shuffle=True,
# )  # shuffle the data?

# test_dataloader_custom = DataLoader(
#     dataset=test_data_custom,  # use custom created test Dataset
#     batch_size=1,
#     num_workers=0,
#     shuffle=False,
# )  # don't usually need to shuffle testing data

# train_dataloader_custom, test_dataloader_custom
