import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor

# Initialize the dataset
coco_dataset = CocoDetection(root = 'path_to_images', 
                             annFile = 'path_to_annotation',
                             transform=ToTensor())

# Initialize the dataloader
data_loader = torch.utils.data.DataLoader(coco_dataset, batch_size=4, shuffle=True, num_workers=2)#output: (images, targets)


