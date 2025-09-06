import torch
import io

# This adds type hints and checking to our data
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Depends
from app.model import load_model, load_transforms
from torchvision.models import ResNet
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F

from PIL import Image


CATEGORIES = [
    "freshapple",
    "freshbanana",
    "freshorange",
    "rottenapple",
    "rottenbanana",
    "rottenorange",
]


# this is a datamodel (datatype same dataclass, but pydantic is implemented in RUST) that describes the output of the API
class Result(BaseModel):
    category: str
    confidence: float


# Create the FastAPI instance
app = FastAPI()


# Debug message to check the server app is running
@app.get("/")
def read_root():
    return {"message": "API is running. Visit /docs for the Swagger API documentation."}


# Response model is the Result dataclass defined above
@app.post("/predict", response_model=Result)
async def predict(
    input_image: UploadFile = File(...),  # ... is a placeholder for the file
    model: ResNet = Depends(
        load_model
    ),  # Dependency because it is an async function, will not execute until load_model is done
    transforms: transforms.Compose = Depends(load_transforms),
) -> Result:
    image = Image.open(
        io.BytesIO(await input_image.read())
    )  # await is needed because it is an async function

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply the transformation Add the batch dimension for inference
    image = transforms(image).unsqueeze(0)  # Unsqeeze adds a batch dimension at dim 0

    # We can also do unsqueeze with reshape(1, 3, 224, 224) but unsqueeze is more elegant
    # image = transforms(image).reshape(1, 3, 224, 224)

    # Inference mode is a context manager that disables gradient calculation and reduces memory consumption
    # We already did model.eval() in load_model. This sets batch
    # norm according to the training stats and turns off dropout.
    with torch.inference_mode():
        output = model(image)  # Forward pass
        category = CATEGORIES[output.argmax()]  # Get the index of the highest score
        confidence = F.softmax(output, dim=1).max().item()
        return Result(category=category, confidence=confidence)
