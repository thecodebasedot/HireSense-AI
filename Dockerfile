# HireSense AI — containerised REST API
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Generate data and train the model at build time so the image ships ready
RUN python src/generate_data.py && python src/train.py

EXPOSE 8000

# Serve the FastAPI app
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]
