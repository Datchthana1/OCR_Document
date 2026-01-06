FROM python:3.11-slim

WORKDIR /OCR

RUN apt-get update && apt-get install -y \
    git \
    poppler-utils \
    poppler-data \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

EXPOSE 9001

CMD ["python", "-m", "uvicorn", "OCR-easyOCR:app", "--host", "0.0.0.0", "--port", "9001", "--reload"]
