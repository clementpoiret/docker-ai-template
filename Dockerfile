FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install base utilities
RUN apt-get update && apt-get install -y apt-transport-https

# For GitPython
RUN apt-get -y update
RUN apt-get -y install git

# /workspace for OVH
RUN mkdir -p /workspace && chown -R 42420:42420 /workspace
ENV HOME /workspace
WORKDIR /workspace

# Fix torch error
RUN mkdir -p /workspace/.cache && chmod -R 777 /workspace/.cache

# Setup ENV
# Install timm
RUN pip install --pre timm

ENV HOME=${HOME} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.4.1

# Fix pydantic error
RUN pip install "poetry==$POETRY_VERSION"

# Install project
COPY pyproject.toml /workspace
RUN POETRY_VIRTUALENVS_CREATE=false \
    poetry install --no-interaction --no-ansi
# RUN python -m pip install -e .

# Temporary fix for pydantic
RUN pip uninstall -y starlette fastapi pydantic
RUN pip install starlette fastapi pydantic

# Copy all other files
COPY . .

# Run project
# CMD ["python", "train.py"]
