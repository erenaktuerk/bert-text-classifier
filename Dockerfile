# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]