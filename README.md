# MLOps-DSR

A small example project that downloads a trained model artifact from Weights & Biases and stores it locally.

## What this repo does

- Uses a small script in `app/model.py` to download a model artifact from W&B.
- Stores the downloaded artifact in a local `models` directory (configurable in `app/model.py`).

## Prerequisites

- Python 3.10+ installed

# MLOps-DSR

Small FastAPI service that loads a trained ResNet model from a W&B artifact and exposes a single prediction endpoint.

## Project structure (important files)

- `app/main.py` — FastAPI app; endpoints: `GET /` (health) and `POST /predict` (image upload -> classification).
- `app/model.py` — model loading, artifact download, and image transforms.
- `requirements.txt` — Python dependencies.

## Prerequisites

- Python 3.10+
- A W&B account and an API key (if you want to download the model artifact)

## Quick start (recommended)

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file at the repository root with these values (replace with your values):

   ```text
   WANDB_API_KEY=your_wandb_api_key
   WANDB_ORG=your_wandb_organization
   WANDB_PROJECT=your_wandb_project
   WANDB_MODEL_NAME=artifact_name
   WANDB_MODEL_VERSION=version_or_alias
   ```

## Run the FastAPI app

Use the command you provided to run the app locally (port 8080, auto-reload):

```bash
fastapi run app/main.py --port 8080 --reload
```

Alternative (common):

```bash
uvicorn app.main:app --port 8080 --reload
```

Open <http://localhost:8080/docs> to view the interactive Swagger UI.

## API

- GET /
  - Health check. Returns a simple JSON message.

- POST /predict
  - Accepts a multipart/form-data file field named `input_image`.
  - Response model:

    ```json
    {
      "category": "freshapple",
      "confidence": 0.92
    }
    ```

  - Example curl (replace image path):

    ```bash
    curl -X POST "http://localhost:8080/predict" -F "input_image=@/path/to/image.jpg"
    ```

## Notes about model & transforms

- `app/model.py` currently sets `MODELS_DIR = "../models"`. That path is resolved relative to the process current working directory — so running the server from the repository root will place `models/` one level above the repo. Consider changing `MODELS_DIR` to be relative to the `app/` directory if you want the models folder created inside the repo. Example:

```python
import os
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)
```

- The default transforms in `load_transforms()` resize the shorter side to 256 and then center-crop to 224, so the network input is 224×224. If you want a final 256×256 image, change the pipeline to `Resize((256,256))` or `Resize(256)` + `CenterCrop(256)`.

## Troubleshooting

- If the models folder is not created: print `MODELS_DIR` in `app/model.py` and check permissions.
- If artifact download fails: verify `.env` variables and that `WANDB_API_KEY` is correct.
- If the server doesn't start: ensure `fastapi`/`uvicorn` is installed and you activated the virtualenv.

## Examples & testing

- Try the Swagger UI at `/docs` for quick manual tests.
- Use the curl example above to test the `/predict` endpoint from the command line.

## License

MIT
