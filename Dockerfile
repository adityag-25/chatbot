FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 7860

# Launch FastAPI — serves both API and static UI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
