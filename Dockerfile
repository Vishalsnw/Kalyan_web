# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port for Flask
EXPOSE 8080

# Set environment variable (Flask will use this port on Cloud Run)
ENV PORT 8080

# Run the Flask app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
