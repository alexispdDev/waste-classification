# Start from a lightweight official Python 3.11 image based on Debian Bookworm
FROM python:3.11.14-slim-bookworm

# Copy the `uv` and `uvx` binaries from the official Astral UV image into /bin/
# This gives access to the fast dependency manager (like pip, but faster and modern)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container
# All subsequent commands will run inside /app
WORKDIR /app

# Add the local virtual environment’s `bin` folder to PATH
# so installed tools and packages can be executed directly
ENV PATH="/app/.venv/bin:$PATH"

# Copy Python project metadata and lock files into the image
# - pyproject.toml: defines project dependencies and build system
# - uv.lock: pinned dependency versions for reproducible installs
# - .python-version: specifies the Python version used locally (for consistency)
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Create folder where to place the application
RUN mkdir -p /app/src/waste_classification

# Copy your application into the container
COPY src/waste_classification/__init__.py src/waste_classification/serve.py src/waste_classification/predict.py /app/src/waste_classification/

# Install dependencies using `uv`, respecting locked versions for consistency
RUN uv sync --no-dev --locked && uv pip install .

# Copy your model file into the container
COPY model/waste_classifier_mobilenet_v4.onnx* /app

# Let the API know where the model is located.
ENV WASTE_CLASSIFICATION_MODEL_PATH="/app/waste_classifier_mobilenet_v4.onnx"

# Expose port 8080 so it can be accessed outside the container
EXPOSE 8080

# Define the command that runs when the container starts
# Here, it runs the serve module.
ENTRYPOINT ["uvicorn", "waste_classification.serve:app", "--host", "0.0.0.0", "--port", "8080"]