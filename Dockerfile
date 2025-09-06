# Set python version for the base image
FROM python:3.11-slim

# set the working directory in the container
WORKDIR /code

# Copy the requirements file into the working directory
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies and packages in the requirements file
RUN pip install -r /code/requirements.txt

# Copy the entire project into the working directory
COPY ./app /code/app

ENV WANDB_API_KEY=""
ENV WANDB_ORG=""
ENV WANDB_PROJECT=""
ENV WANDB_MODEL_NAME=""
ENV WANDB_MODEL_VERSION=""

EXPOSE 8080

CMD ["fastapi", "run", "app/main.py", "--port", "8080", "--reload" ]
