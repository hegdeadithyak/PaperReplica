import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

from going_modular.going_modular import data_setup,engine
from helper_functions import download_data,set_seeds,plot_loss_curves

device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "./data/pizza_steak_sushi"

train_dir = image_path + "/train"
test_dir = image_path + "/test"


IMG_SIZE = 224
BATCH_SIZE = 32

manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

print(f"Manual transforms:\n{manual_transforms}")

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)

image_batch, label_batch = next(iter(train_dataloader)) #next(iter(train_dataloader)) is a generator in which next is used to get the next element in the generator iter is used to convert the object into an iterator

image, label = image_batch[0], label_batch[0]

print(f"Image shape: {image.shape}\nLabel: {label}\nClass name: {class_names[label]}")

# plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()


