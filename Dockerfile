# Use an official lightweight Python image as base
# 'slim' reduces image size by excluding unnecessary libraries and tools
FROM python:3.10-slim

# Set environment variables to avoid Python buffer issues and create a more predictable environment
# Ensures that logs are immediately flushed, no buffering:
ENV PYTHONUNBUFFERED=1 \ 
# Prevents Python from creating .pyc files, saving space                    
    PYTHONDONTWRITEBYTECODE=1 \       
# Adds local Python bin directory to PATH for convenience       
    PATH="/root/.local/bin:$PATH"            

# Set working directory inside the container
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && \
# Avoids installing unnecessary packages
    apt-get install -y --no-install-recommends \ 
    # Required for compiling some Python libraries 
    build-essential \            
    # Cleans up package manager cache to keep image small
    && rm -rf /var/lib/apt/lists/*              

# Copy requirements file and install Python dependencies
# Copies only requirements.txt to leverage Dockers caching
COPY requirements.txt .                 
# Installs Python libraries without caching to reduce image size       
RUN pip install --no-cache-dir -r requirements.txt  

# Copy the entire application code after installing dependencies
# This ensures Docker rebuilds only when app code changes, not on every small code tweak
COPY . .

# Pre-download NLTK data at build time to avoid runtime delays and ensure the container is ready to go
RUN python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger

# Expose FastAPI’s port (8000) so it can be reached from outside the container
EXPOSE 8000

# Start the FastAPI app using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 binds the app to the correct port
# --workers 4 allows handling multiple requests in parallel, improving performance
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]