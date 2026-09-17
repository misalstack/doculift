FROM python:3.10-slim

# System dependencies for PaddleOCR + OpenCV + PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements-space.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY src/ src/
COPY config/ config/
COPY run_app.py run_app.py
COPY pyproject.toml pyproject.toml

# Streamlit config
COPY .streamlit/ .streamlit/

# Create writable data directories
RUN mkdir -p data/uploads data/outputs data/temp

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Railway injects $PORT dynamically; fall back to 8501 for local/other platforms
CMD ["sh", "-c", "python -m streamlit run src/web/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
