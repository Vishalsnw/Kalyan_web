# Use Python 3.11 (fully compatible with current Pandas & Cython builds)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port (Render will inject PORT env var)
ENV PORT 10000

# Run the Flask app
CMD ["python", "app.py"]
