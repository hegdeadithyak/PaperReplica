import matplotlib.pyplot as plt
import torch

from torch import nn
from torchvision import transforms
from going_modular.going_modular import data_setup, engine

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


class MAS(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0
    ):
        super().__init__()

    def forward(self, x):
        x = nn.LayerNorm(x)
        att_out, _ = nn.MultiheadAttention(x, x, x)
        return att_out


class MLP(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, mlp_size: int = 3072, mlp_dropout: float = 0.1
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        attn_dropout: float = 0,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = MAS(embedding_dim, num_heads, attn_dropout)
        self.mlp = MLP(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)

        return x


class Vit(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_dim: int = 768,
        mlp_size: int = 3071,
        num_heads: int = 12,
        attn_dropout: int = 0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )

        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True
        )

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmebedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoder(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    mlp_dropout=mlp_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])  # run on each sample in a batch at 0 index
        return x


vit = Vit(num_classes=len(class_names))
optimizer = torch.optim.Adam(
    params=vit.parameters(),
    lr=3e-3,  # Base LR from Table 3 for ViT-* ImageNet-1k
    betas=(
        0.9,
        0.999,
    ),
    weight_decay=0.3,
)

loss_fn = torch.nn.CrossEntropyLoss()
results = engine.train(
    model=vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
)
