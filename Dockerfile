# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required packages
RUN pip install --no-cache-dir numpy pandas matplotlib seaborn scikit-learn mlflow

# Run script.py when the container launches
CMD ["python", "app.py"]