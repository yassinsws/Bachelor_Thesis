import json
import os
import time
import fitz  # PyMuPDF
import openai
import weaviate
from weaviate.gql.get import HybridFusion
from unstructured.cleaners.core import clean

azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
resource_name = os.getenv("RESOURCE_NAME")
headers = {
    "X-Azure-Api-Key": azure_openai_key,
}


def prompt(query):
    return f""" You are a university professor.
    Answer the following question using the provided context.
    If you can't find the answer, do not pretend you know it, ask for more information"
    Answer in the same langauge as the question. If you used your own knowledge apart from the context provided mention that.
    Question:  {query} """


def chunk_files(subdirectory_path, subdirectory):
    data = []
    # Process each PDF file in this subdirectory
    for filename in os.listdir(subdirectory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(subdirectory_path, filename)
            str_five = ""
            # Open the PDF
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page_text = doc[page_num].get_text()
                    page_text = clean(page_text, bullets=True, extra_whitespace=True)
                    slide_id = filename + str(page_num)
                    if page_num % 5 == 0:
                        if page_num != 0:  # Avoid appending empty content for the first page
                            data.append({
                                "content": str_five,
                                "slide_id": slide_id,
                                "page_interval": str(str(page_num - 5) + "->" + str(page_num)),
                                "lecture_id": subdirectory  # Save the subdirectory name
                            })
                        last_page = doc[page_num - 1].get_text() if page_num > 0 else ""
                        last_page = clean(last_page, bullets=True, extra_whitespace=True)
                        str_five = last_page + page_text
                    else:
                        str_five += "\n\n" + page_text
                # Append the last accumulated text if it's not empty
                if str_five:
                    data.append({
                        "content": str_five,
                        "slide_id": subdirectory_path + str(len(doc)),
                        "page_interval": str(str(len(doc) - 10) + "->" + str(len(doc))),
                        "lecture_id": subdirectory  # Save the subdirectory name
                    })
    return data


class AI:
    def __init__(self):
        api_key_header = {
            "X-Azure-Api-Key": azure_openai_key,  # Replace with your inference API key
        }
        self.client = weaviate.Client(
            url="http://localhost:8080",  # Replace with your endpoint
            additional_headers=api_key_header
        )

    def create_class(self):
        t2v = {
            "model": "ada",
            "modelVersion": "002",
            "type": "text",
            "baseURL": azure_endpoint,
            "resourceName": resource_name,
            "deploymentId": "te-ada-002",
        }
        self.client.schema.delete_class("Lectures")
        if not self.client.schema.exists("Lectures"):
            class_obj = {
                "class": "Lectures",
                "vectorizer": "text2vec-openai",
                "properties": [
                    {
                        "dataType": ["text"],
                        "name": "content",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "vectorizePropertyName": False
                            }
                        },
                    },
                    {
                        "dataType": ["text"],
                        "name": "slide_id",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "vectorizePropertyName": False
                            }
                        },
                    },
                    {
                        "dataType": ["text"],
                        "name": "page_interval",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "vectorizePropertyName": False
                            }
                        },
                    },
                    {
                        "dataType": ["text"],
                        "name": "lecture_id",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "vectorizePropertyName": False
                            }
                        },
                    }
                ],
                "moduleConfig": {
                    "text2vec-openai": t2v,
                    "generative-openai": {
                        "baseURL": azure_endpoint,
                        "resourceName": resource_name,
                        "deploymentId": "gpt-35-16k",
                        "waitForModel": True,
                        "useGPU": False,
                        "useCache": True
                    }
                },
            }
            self.client.schema.create_class(class_obj)
            print("Schema created")
            directory_path = "../../lectures"
            print("Importing data into the batch")
            # Iterate through each subdirectory in the root directory
            for subdirectory in os.listdir(directory_path):
                subdirectory_path = os.path.join(directory_path, subdirectory)
                if os.path.isdir(subdirectory_path):
                    self.batch_import(subdirectory_path, subdirectory)
            print("Import Finished")

    def batch_import(self, directory_path, subdirectory):
        data = chunk_files(directory_path, subdirectory)
        # Configure a batch process
        self.client.batch.configure(
        # `batch_size` takes an `int` value to enable auto-batching
        # dynamically update the `batch_size` based on import speed
                dynamic=True,
                timeout_retries=0
        )
        with self.client.batch as batch:
            # Batch import all Questions

            for i, d in enumerate(data):
                embeddings_created = False
                properties = {
                    "content": d["content"],
                    "slide_id": d["slide_id"],
                    "page_interval": d["page_interval"],
                    "lecture_id": d["lecture_id"]
                }
                # Initialize the flag
                embeddings_created = False
                # create embeddings (exponential backoff to avoid RateLimitError)
                for j in range(5):  # max 5 retries
                    # Only attempt to create embeddings if not already created
                    if not embeddings_created:
                        try:
                            batch.add_data_object(
                                properties,
                                "Lectures"
                            )
                            embeddings_created = True  # Set flag to True on success
                            break  # Break the loop as embedding creation was successful
                        except openai.error.RateLimitError:
                            time.sleep(2 ** j)  # wait 2^j seconds before retrying
                            print("Retrying import...")
                    else:
                        break  # Exit loop if embeddings already created

                # Raise an error if embeddings were not created after retries
                if not embeddings_created:
                    raise RuntimeError("Failed to create embeddings.")

    def generate_response(self, query, lecture_id):
        if lecture_id != "" and lecture_id is not None:
            response = (
                self.client.query
                .get("Lectures", ["content", "slide_id", "page_interval", "lecture_id"])
                .with_where({
                    "path": ["lecture_id"],
                    "operator": "Equal",
                    "valueText": lecture_id
                })
                .with_near_text({"concepts": query})
                .with_additional(f'rerank( query: "{query}", property: "content"){{score}}')
                .with_generate(grouped_task=prompt(query))
                .with_limit(3)
                .do()
            )
            generated_response = response["data"]["Get"]["Lectures"][2]["_additional"]["generate"]["groupedResult"]

        else:
            response = (
                self.client.query
                .get("Lectures", ["content", "slide_id", "page_interval", "lecture_id"])
                # alpha = 0 forces using a pure keyword search method (BM25)
                # alpha = 1 forces using a pure vector search method
                .with_hybrid(query=query,
                             alpha=1,
                             fusion_type=HybridFusion.RELATIVE_SCORE
                             )
                # .with_additional(f'rerank( query: "{query}", property: "content"){{score}}')
                .with_generate(grouped_task=prompt(query))
                .with_limit(3)
                .do()
            )
            generated_response = response["data"]["Get"]["Lectures"][0]["_additional"]["generate"]["groupedResult"]
            print(json.dumps(response, indent=2))
        return generated_response