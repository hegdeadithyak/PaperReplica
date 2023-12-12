import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__all__ = ['CustomDataset']


# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def find_classes(directory: str)-> Tuple[List[str], Dict[str, int]]:
    classes =sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't find any class in {directory}.")
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



class CustomDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes,self.class_to_idx = find_classes(self.data_dir) #find classes and class_to_idx

    def load_image(self,index:int)->Image.Image:
        # This should be changed as per your dataset structure
        # image_path = os.path.join(self.data_dir,self.classes[index//100],f'{index%100}.jpg')
        return Image.open(image_path)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        image =self.load_image(index)
        label = self.class_to_idx[image_path.parent.name]

        if self.transform:
            image = self.transform(image)
            return image, label
        else:
            return image, label

