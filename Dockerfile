FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY config/ config/
COPY src/ src/
COPY artifacts/ artifacts/
COPY app/ app/
COPY data/ data/

# Expose ports
EXPOSE 8000 8501

# Default command: start API
CMD ["uvicorn", "src.sentinel.api:app", "--host", "0.0.0.0", "--port", "8000"]