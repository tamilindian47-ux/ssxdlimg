FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image. Override with --build-arg MODEL_ID=owner/repo
ARG MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
ENV MODEL_ID=${MODEL_ID}
RUN python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['MODEL_ID'], allow_patterns=['*.json','*.txt','*fp16*.safetensors'])"

COPY handler.py .
CMD ["python", "-u", "handler.py"]
