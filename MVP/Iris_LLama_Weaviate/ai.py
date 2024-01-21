import json

import fitz  # PyMuPDF
import llama_index
import openai
import weaviate
from llama_index.node_parser import SimpleNodeParser

from unstructured.cleaners.core import clean
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, ServiceContext, load_index_from_storage, StorageContext
import pickle
import os
from llama_index import download_loader, VectorStoreIndex
from llama_hub.github_repo import GithubClient, GithubRepositoryReader

azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
resource_name = os.getenv("RESOURCE_NAME")
azure_client = openai.lib.azure.AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15"
)
headers = {
    "X-Azure-Api-Key": azure_openai_key,
}

def query_openai(messages):
    return azure_client.chat.completions.create(
        model="gpt-35-16k",  # model = "deployment_name".
        messages=messages
    )


def prompt(query):
    return f""" You are a university professor.
    Answer the following question using only the provided context.
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
        self.index = None
        self.query_engine = None
        api_key_header = {
            "X-Azure-Api-Key": azure_openai_key,  # Replace with your inference API key
        }
        self.client = weaviate.Client(
            url="http://localhost:8080",  # Replace with your endpoint
            additional_headers=api_key_header
        )
        self.docs = None

    def create_class_weaviate(self):
        t2v = {
            "model": "ada",
            "modelVersion": "002",
            "type": "text",
            "baseURL": azure_endpoint,
            "resourceName": resource_name,
            "deploymentId": "te-ada-002",
        }
        self.client.schema.delete_class("Repo")
        if not self.client.schema.exists("Repo"):
            class_obj = {
               "class": "Repo",
                "vectorizer": "text2vec-openai",
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
                }
            }
            self.client.schema.create_class(class_obj)
            print("Schema created")

    def create_class_llama(self):
        download_loader("GithubRepositoryReader")
        llm = llama_index.llms.AzureOpenAI(model="gpt-35-turbo-16k", deployment_name="gpt-35-16k",
                                           api_key=azure_openai_key, azure_endpoint=azure_endpoint,
                                           api_version="2023-03-15-preview")
        embed_model = llama_index.embeddings.AzureOpenAIEmbedding(
            model="text-embedding-ada-002",
            deployment_name="te-ada-002",
            api_key=azure_openai_key,
            azure_endpoint=azure_endpoint,
            api_version="2023-03-15-preview"
        )
        if self.docs is None:
            github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
            loader = GithubRepositoryReader(
                github_client,
                owner="yassinsws",
                repo="Tum_AI_Assistant",
                verbose=True,
                concurrent_requests=10,
                timeout=5,
            )
            self.docs = loader.load_data(branch="main")
            with open("docs.pkl", "wb") as f:
                pickle.dump(self.docs, f)
        self.create_class_weaviate()
        vector_store = WeaviateVectorStore(weaviate_client=self.client, index_name="Repo", text_key="content")
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(self.docs, storage_context=storage_context, service_context=service_context)
        self.query_engine = self.index.as_query_engine()

    def generate_response(self, user_message, lecture_id):
        if self.query_engine is None:
            vector_store = WeaviateVectorStore(
                weaviate_client=self.client, index_name="Repo"
            )
            loaded_index = VectorStoreIndex.from_vector_store(vector_store)
            query_engine = loaded_index.as_query_engine(
                vector_store_query_mode="hybrid", similarity_top_k=3, alpha=1
            )
            response = query_engine.query(user_message)
            print(response.__str__())
            return response.__str__()
