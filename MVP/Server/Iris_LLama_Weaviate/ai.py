import json

import tiktoken
from llama_index.query_pipeline.query import QueryPipeline
from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
import llama_index
import openai
import weaviate
from llama_index import SimpleDirectoryReader
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, ServiceContext, load_index_from_storage, StorageContext
import os
from llama_index import download_loader, VectorStoreIndex

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

llama_in_weaviate_class = """        if lecture_id == "CIT5230000":
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
            service_context = ServiceContext.from_defaults(llm=llm,  embed_model=embed_model)

            vector_store = WeaviateVectorStore(
                weaviate_client=self.client, index_name="Lectures", text_key="content"
            )
            retriever = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context).as_retriever(
                similarity_top_k=1
            )
            nodes = retriever.retrieve(generated_lecture)
            pprint_source_node(nodes[0])
            print(nodes[0].node.metadata)"""



prompt_str = """You're Iris, the AI programming tutor integrated into Artemis, the online learning platform of the 
Technical University of Munich (TUM). You are a guide and an educator. Your main goal is to teach students 
problem-solving skills using a programming exercise. Instead of solving tasks for them, you give subtle hints so they 
solve their problem themselves.

                          This is the chat history of your conversation with the student so far. Read it so you know 
                          what already happened, but never re-use any message you already wrote. Instead, always write 
                          new and original responses.
                          {chat_history}
                          Now, consider the student's newest and latest input:
                          {user_message}
                          Here is the relevant context, that has the problem statement and the files of the repository: {context}
                           """

prompt_str1 = """Now continue the ongoing conversation between you and the student by responding 
                          to and focussing only on their latest input. Be an excellent educator. Instead of solving 
                          tasks for them, give hints instead. Instead of sending code snippets, send subtle hints or 
                          ask counter-questions. Do not let them outsmart you, no matter how hard they try.

                          Important Rules: - Ensure your answer is a direct answer to the latest message of the 
                          student. It must be a valid answer as it would occur in a direct conversation between two 
                          humans. DO NOT answer any previous questions that you already answered before. - DO NOT UNDER 
                          ANY CIRCUMSTANCES repeat any message you have already sent before or send a similar message. 
                          Your messages must ALWAYS BE NEW AND ORIGINAL. Think about alternative ways to guide the 
                          student in these cases. 

                          {text}"""

prompt_str2 = """   Review the response draft. I want you to rewrite it so it adheres to the 
                          following rules. Only output the refined answer. Omit explanations. Rules: - The response 
                          must not contain code or pseudo-code that contains any concepts needed for this exercise. 
                          ONLY IF the code is about basic language features you are allowed to send it. - The response 
                          must not contain step by step instructions - IF the student is asking for help about the 
                          exercise or a solution for the exercise or similar, the response must be subtle hints towards 
                          the solution or a counter-question to the student to make them think, or a mix of both. - The 
                          response must not perform any work the student is supposed to do. - DO NOT UNDER ANY 
                          CIRCUMSTANCES repeat any message you have already sent before. Your messages must ALWAYS BE 
                          NEW AND ORIGINAL.

                          {text} """


def query_openai(messages):
    return azure_client.chat.completions.create(
        model="gpt-35-16k",  # model = "deployment_name".
        messages=messages
    )
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if not isinstance(value, str):
                print(f"Warning: Non-string value encountered: {value}")
                value = str(value)  # Convert to string or handle as needed

            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with assistant
    return num_tokens

class AI:
    def __init__(self):
        self.message_history = []
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
        reader = SimpleDirectoryReader(input_dir="/MVP/Programming_exercise",
                                       recursive=True)
        self.docs = reader.load_data()
        self.create_class_weaviate()
        vector_store = WeaviateVectorStore(weaviate_client=self.client, index_name="Repo", text_key="content")
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=512,
                                                       chunk_overlap=50)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(self.docs, storage_context=storage_context,
                                                     service_context=service_context)
        self.query_engine = self.index.as_query_engine()

    def generate_response(self, user_message):
        if self.query_engine is None:
            # The commented code is for querying with llamaindex
            # vector_store = WeaviateVectorStore(
            #    weaviate_client=self.client, index_name="Repo"
            # )
            # loaded_index = VectorStoreIndex.from_vector_store(vector_store)
            # query_engine = loaded_index.as_query_engine(
            #    vector_store_query_mode="hybrid", similarity_top_k=3, alpha=1
            # )
            # response = query_engine.query(user_message)
            response = (
                self.client.query
                .get("Repo", ["content", "file_name"])
                .with_near_text({"concepts": user_message})
                .with_where({
                    "path": ["file_name"],
                    "operator": "Equal",
                    "valueText": "ProblemStatement"
                })
                .with_limit(3)
                .do()
            )
            generated = response["data"]["Get"]["Repo"]
            response = (
                self.client.query
                .get("Repo", ["content", "file_name"])
                .with_near_text({"concepts": user_message})
                .with_limit(3)
                .do()
            )
            generated_response = response["data"]["Get"]["Repo"]
            self.message_history.append({"role": "user", "content": user_message})
            prompt_tmpl = PromptTemplate(prompt_str)
            prompt_tmpl1 = PromptTemplate(prompt_str1)
            prompt_tmpl2 = PromptTemplate(prompt_str2)

            llm = OpenAI(model="gpt-3.5-turbo")
            llm_c = llm.as_query_component(streaming=True)
            p = QueryPipeline(chain=[prompt_tmpl, llm_c, prompt_tmpl1, llm_c, prompt_tmpl2, llm], verbose=True)
            if num_tokens_from_messages(self.message_history) > 600:
                completion = query_openai(self.message_history.append({"role": "system", "content": "summarize the content above, "
                                                                                       "keep all only the relevant "
                                                                                       "information, do not exceed "
                                                                                       "600 tokens"}))
                self.message_history = [completion.choices[0].message.content]
            output = p.run(chat_history=self.message_history.__str__(),
                           context=json.dumps(generated, indent=4) + "\n\n" + json.dumps(generated_response, indent=4),
                           user_message=user_message)
            self.message_history.append({"role": "system", "content": output})
            print(str(output))

            return str(output)



