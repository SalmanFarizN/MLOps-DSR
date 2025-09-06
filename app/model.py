import os
import wandb

# This gives us access to the variables set in the .env file
from loadotenv import load_env
from torchvision.models import resnet18, ResNet
from torch import nn
from pathlib import Path
import torch
from torchvision.transforms import v2 as transforms

load_env()
wandb_api_key = os.environ.get("WANDB_API_KEY")

# Local folder to store models
MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"
os.makedirs(MODELS_DIR, exist_ok=True)


def download_artifact():
    """Download the model weights from Weights & Biases."""
    assert (
        "WANDB_API_KEY" in os.environ
    ), "WANDB_API_KEY not found in environment variables. Please set it in the .env file."
    wandb.login(key=wandb_api_key)
    api = wandb.Api()

    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")

    artifact_path = (
        f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"
    )
    print(f"Downloading artifact from {artifact_path}")

    artifact = api.artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)


def get_raw_model() -> ResNet:
    """Get the architecture of the model (random weights), this must match the architecture of the trained model."""
    architecture = resnet18(weights=None)

    architecture.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=6),
    )

    return architecture


def load_model() -> ResNet:
    """Loads the model and adds the trained weights from the .pth file."""
    download_artifact()
    model = get_raw_model()

    model_state_dict_path = Path(MODELS_DIR) / MODEL_FILENAME
    # This loads the trained weights from the .pth file
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")

    # This merges the trained weights into the model architecture
    model.load_state_dict(model_state_dict, strict=True)

    # Set the model to evaluation mode
    # Turn off dropout. Batch normalization layers uses stats from training
    # (fine-tuning we did)
    model.eval()

    return model


live_resnet = load_model()


def load_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            # Resize the image so that the shorter side is 256 pixels (often 256x256)
            transforms.Resize(256),
            # Then take a center crop of 224x224
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
