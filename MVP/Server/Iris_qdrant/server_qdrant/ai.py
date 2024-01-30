import os
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.http import models
from unstructured.cleaners.core import clean
from openai.lib.azure import AzureOpenAI


def chunk_files(subdirectory_path, subdirectory):
    vector_data = []
    payload_data = []
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
                            vector_data.append(str_five)
                            payload_data.append({
                                "slides_content": str_five,
                                "slide_id": slide_id,  # Example: ITP2324 L01 Introduction.pdf 66
                                "lecture_id": subdirectory,  # Example: CIT5230000
                                "page_interval": str(str(page_num - 5) + "->" + str(page_num))
                            })
                        last_page = doc[page_num - 1].get_text() if page_num > 0 else ""
                        last_page = clean(last_page, bullets=True, extra_whitespace=True)
                        str_five = last_page + page_text
                    else:
                        str_five += "\n\n" + page_text
                # Append the last accumulated text if it's not empty
                if str_five:
                    vector_data.append(str_five)
                    payload_data.append({
                        "slides_content": str_five,
                        "slide_id": subdirectory_path + str(len(doc)),
                        "lecture_id": subdirectory,  # Save the subdirectory name
                        "page_interval": str(str(len(doc) - 10) + "->" + str(len(doc)))
                    })
    return vector_data, payload_data


class AI:
    def __init__(self):
        self.azure_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15"
        )
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "lectures"
#        self.collection = self.client.recreate_collection(
#            collection_name=self.collection_name,
#            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
#        )
        directory_path = "../../../lectures"
        print("Importing data into the batch")
        # Iterate through each subdirectory in the root directory
#        for subdirectory in os.listdir(directory_path):
#            subdirectory_path = os.path.join(directory_path, subdirectory)
#            if os.path.isdir(subdirectory_path):
#                self.batch_import(subdirectory_path, subdirectory)
        print("Import Finished")

    def query_openai(self, messages):
        return self.azure_client.chat.completions.create(
            model="gpt-35-16k",  # model = "deployment_name".
            messages=messages
        )

    def get_embedding(self, text, model="te-ada-002"):  # model = "deployment_name"
        response = self.azure_client.embeddings.create(
            input=text,
            model=model)
        embeddings = [embedding.embedding for embedding in response.data]
        return embeddings

    def batch_import(self, directory_path, subdirectory):
        vector_data, payload_data = chunk_files(directory_path, subdirectory)
        data = self.get_embedding(vector_data)
        index = list(range(len(data)))
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=index,
                # Qdrant can only take in native Python iterables like lists and tuples.
                vectors=data,
                payloads=payload_data
            )
        )

    def generate_response(self, question: str) -> str:
        question_vector = self.get_embedding(question)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=question_vector[0],
            limit=3,
        )
        print(results)
        context = "\n".join(r.payload.get("slides_content") for r in results)
        completion = self.query_openai(
            messages=[
                {"role": "user",
                 "content": f""" You are a university professor.
                  Answer the following question using the provided context.
                   If you can't find the answer, do not pretend you know it.
                   Rather ask for more information
                   Answer in the same langauge as the question.
                    If you used your own knowledge apart from the context provided mention that.
\n
                   
                   Question: {question.strip()}\n
                    
                    Context: {context.strip()}
                    Answer:"""}])
        return completion.choices[0].message.content
