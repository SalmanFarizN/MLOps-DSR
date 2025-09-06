import torch
import io

# This adds type hints and checking to our data
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Depends
from app.model import load_model, load_transforms
from torchvision.models import ResNet
from torchvision.transforms import v2 as transforms


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
    transforms: transforms.Compose = Depends(
        load_transforms
    ),  # Dependency because it is an async function, will not execute until load_transforms is done
):  # missing the return type
    pass
