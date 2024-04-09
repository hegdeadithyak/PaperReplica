import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from going_modular.going_modular import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

image_path = "VisionTransformer/data/pizza_steak_sushi"

train_dir = image_path + "/train"
test_dir = image_path + "/test"


IMG_SIZE = 224
BATCH_SIZE = 32

manual_transforms = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)

print(f"Manual transforms:\n{manual_transforms}")

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,  # use manually created transforms
    batch_size=BATCH_SIZE,
)

image_batch, label_batch = next(
    iter(train_dataloader)
)  # next(iter(train_dataloader)) is a generator in which next is used to get the next element in the generator iter is used to convert the object into an iterator

image, label = image_batch[0], label_batch[0]

print(f"Image shape: {image.shape}\nLabel: {label}\nClass name: {class_names[label]}")

# plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

# To - do :

# Patch Embeding --> done
# MSA (Multi Self Attention)
# MLP (Multi Layer Perceptron)
# Layer Norm


class PatchEmebedding(nn.Module):
    """
    changes 2D input image into 1D embedded image

    Args :
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768. --> 16 x 16 x 3
    """

    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ):
        super().__init__()

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        self.flatten = nn.Flatten(
            start_dim=2, end_dim=3
        )  # height and width are flattend to 1D

    def forward(self, x):
        image_resolution = x.shape[-1]  # 224 in this case
        assert (
            image_resolution % 16 == 0
        ), f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {16}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1)


patchiee = PatchEmebedding()

# image = image.unsqueeze(0)
# print(f"Input image shape: {image.shape}")
# patch_embedded_image = patchiee(image)
# print(f"Output patch embedding shape: {patch_embedded_image.shape}")

# random_input_image = (1, 3, 224, 224)
# print(summary(PatchEmebedding(),
#         input_size=random_input_image, # try swapping this for "random_input_image_error"
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]))


class Vit(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_num: int = 768,
        mlp_size: int = 3071,
        num_heads: int = 12,
        attn_dropout: int = 0,
        mlp_dorpout: float = 0.1,
        embedding_dropout: float = 0.1,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
