import os
import wandb

# This gives us access to the variables set in the .env file
from loadotenv import load_env

load_env()
wandb_api_key = os.environ.get("WANDB_API_KEY")

# Local folder to store models
MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"
os.makedirs(MODELS_DIR, exist_ok=True)


def download_artifact():
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


download_artifact()
