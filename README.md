# MLOps-DSR

A small example project that downloads a trained model artifact from Weights & Biases and stores it locally.

## What this repo does

- Uses a small script in `app/model.py` to download a model artifact from W&B.
- Stores the downloaded artifact in a local `models` directory (configurable in `app/model.py`).

## Prerequisites

- Python 3.10+ installed

# MLOps-DSR

A small example project that downloads a trained model artifact from Weights & Biases and stores it locally.

## What this repo does

- Uses a small script in `app/model.py` to download a model artifact from W&B.
- Stores the downloaded artifact in a local `models` directory (configurable in `app/model.py`).

## Prerequisites

- Python 3.10+ installed
- A W&B account and an API key

## Setup

1. Create and activate a virtual environment (recommended):

 ```bash
 python -m venv .venv
 source .venv/bin/activate
 ```

2. Install dependencies:

 ```bash
 pip install -r requirements.txt
 ```

3. Add a `.env` file at the repository root with the following variables (replace values):

 ```text
 WANDB_API_KEY=your_wandb_api_key
 WANDB_ORG=your_wandb_organization
 WANDB_PROJECT=your_wandb_project
 WANDB_MODEL_NAME=artifact_name
 WANDB_MODEL_VERSION=version_or_alias
 ```

The project expects these environment variables to be loaded at runtime (the code calls `load_env()` in `app/model.py`).

## Usage

From the repository root run:

```bash
python -m app.model
```

This will call the `download_artifact()` function in `app/model.py`, log into W&B using `WANDB_API_KEY`, and download the specified artifact to the configured models directory.

## Notes about the models directory

- The script currently defines `MODELS_DIR = "../models"` inside `app/model.py`. That path is interpreted relative to the current working directory when the script runs. If you run from the repository root, `../models` resolves to one level above the repo (which may be unexpected).

- Recommended options:

  - Run the script from inside the `app/` directory so `../models` lands inside the repo (not ideal).

  - Or update `app/model.py` to create the models folder relative to the script location (recommended). Example change:

  ```python
  import os
  MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
  os.makedirs(MODELS_DIR, exist_ok=True)
  ```

  This ensures `models/` is created inside the repository regardless of the current working directory.

## Troubleshooting

- If the folder isn't created:

  - Check that your `.env` is loaded and that the script runs to completion.
  - Verify file system permissions for the parent directory.
  - Print `MODELS_DIR` in `app/model.py` before calling `os.makedirs` to see the resolved path.

- If W&B login fails, ensure `WANDB_API_KEY` is correct and that `wandb` is installed.

## Next steps (optional)

- Make `MODELS_DIR` configurable via an environment variable.
- Add basic unit tests and a small CI workflow to validate the download logic.

## License

MIT
