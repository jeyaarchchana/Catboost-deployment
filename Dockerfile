# Base image with Python
FROM python:3.10-slim

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Copy all project files
COPY . . 
#COPY ./app ./app
#COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip 

RUN pip install -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Run the app using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
