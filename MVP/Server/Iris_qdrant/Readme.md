# Iris Qdrant

This MVP explores the use of the vector-database Qdrant to create a chatbot. The chatbot aims to answer questions from students and explain concepts from lectures.

## Setup

**Prerequisite:**
- Docker must be installed on your system. If you haven't installed it yet, you can download it from [Docker's official website](https://www.docker.com/get-started).

## How to Run

1. **Navigate to the Directory:**
   - Go to the `Iris_qdrant/server_qdrant` directory.

2. **Install Python Dependencies:**
   - Install the required Python packages by running:
     ```
     pip install -r requirements.txt
     ```

3. **Pull and Run the Qdrant Docker Image:**
   - Pull the Qdrant Docker image:
     ```
     docker pull qdrant/qdrant
     ```
   - Start the Qdrant service:
     ```
     docker run -p 6333:6333 qdrant/qdrant
     ```

4. **Set Environment Variables:**
   - Configure the following environment variables with their respective values:
     ```
     AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
     AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY
     RESOURCE_NAME=YOUR_RESOURCE_NAME
     ```
