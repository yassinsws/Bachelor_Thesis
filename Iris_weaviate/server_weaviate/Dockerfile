# Use Python 3.10 image as the base image
FROM python:3.10

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app

# Install the Python dependencies defined in requirements.txt
RUN pip3 install -r requirements.txt

# Copy the Python scripts into the container
COPY app.py /app/app.py
COPY ai.py /app/ai.py

# Expose port 5000 to the host machine
EXPOSE 5000

# Set the command to run the app.py script when the container starts
CMD ["python3", "app.py"]
