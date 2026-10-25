FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only torch keeps the image small; swap the index URL for a CUDA build.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY models ./models
COPY scripts ./scripts

# KITTI data is mounted, not baked in:
#   docker run --rm -v $PWD/data:/app/data -v $PWD/output:/app/output lanedet python src/main.py --profile
CMD ["python", "-m", "pytest", "-q", "tests"]
