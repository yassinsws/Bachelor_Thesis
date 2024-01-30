# Iris Weaviate

This is an MVP (Minimum Viable Product) to try out the vector-database Weaviate for creating a chatbot. This chatbot is designed to answer questions from students and explain concepts from lectures.

## How to Run

**Prerequisites:**
- Ensure Docker is installed on your system. If not, you can install it from [Docker's official website](https://www.docker.com/get-started).

**Steps:**

1. **Check for `docker-compose.yml` File:**
   - Make sure you have a `docker-compose.yml` file for setting up the Docker container.
   - If it is not provided, refer to the Weaviate documentation: [Weaviate Docker Compose Overview](https://weaviate.io/developers/weaviate/installation/docker-compose#overview).

2. **Install Python Dependencies:**
   - Run the following command to install the required Python packages:
     ```
     pip install -r requirements.txt
     ```

3. **Start the Docker Container:**
   - Use Docker Compose to set up and start the service:
     ```
     docker-compose up -d
     ```

4. **Set Environment Variables:**
   - Set the following environment variables with their respective values:
     ```
     AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
     AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY
     RESOURCE_NAME=YOUR_RESOURCE_NAME
     ```
     