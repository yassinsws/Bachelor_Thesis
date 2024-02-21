import json
import time
import openai
import weaviate
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
import os
import pickle

azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
resource_name = os.getenv("RESOURCE_NAME")
azure_client = openai.lib.azure.AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15"
)
openai_key = os.getenv("OPENAI_API_KEY")
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


class AI:
    def __init__(self):
        self.combined_elements = None
        self.filtered_elements = None
        api_key_header = {
            "X-Azure-Api-Key": azure_openai_key,  # Replace with your inference API key
            "X-OpenAI-Api-Key": openai_key
        }
        self.client = weaviate.Client(
            url="http://localhost:8080",  # Replace with your endpoint
            additional_headers=api_key_header
        )

    def chunk(self):
        if os.path.exists("raw_pdf_elements.pkl"):
            # Load raw_pdf_elements from the saved file
            with open("raw_pdf_elements.pkl", "rb") as file:
                raw_pdf_elements = pickle.load(file)
        else:
            # Get elements
            raw_pdf_elements = partition_pdf(
                filename="/Users/swss/PycharmProjects/Bachelor_Thesis/MVP/lectures/CIT5230000/ITP2324 L04 Object "
                         "Orientation I.pdf",
                extract_images_in_pdf=True,
                infer_table_structure=True,
                include_metadata=True,
                chunking_strategy="hi_res",
                max_characters=1200,
                new_after_n_chars=600,
                combine_text_under_n_chars=600,
                extract_image_block_to_payload=True
            )
            # Save raw_pdf_elements to a file
            with open("raw_pdf_elements.pkl", "wb") as file:
                pickle.dump(raw_pdf_elements, file)
        raw_pdf_elements = [element.to_dict() for element in raw_pdf_elements]
        self.filtered_elements = [element for element in raw_pdf_elements if
                                  element['type'] not in ['Footer', 'Header', 'UncategorizedText']]

    def combine_elements_by_page(self):
        combined_results = []
        current_combined_text = ""
        current_page_elements = []

        # Initialize with the first element's page number if the list is not empty
        current_page_number = self.filtered_elements[0]['metadata']['page_number'] if self.filtered_elements else None

        for element in self.filtered_elements:
            # Check if the current element is from a new page
            if 'metadata' not in element:
                continue
            if element['metadata']['page_number'] != current_page_number:
                # Append results for the previous page
                if current_page_elements:
                    combined_results.append({'content': current_combined_text, 'elements': current_page_elements})
                    self.filtered_elements.append({'content': current_combined_text, 'page_number': current_page_number})
                # Reset for the new page
                current_combined_text = element['text']
                current_page_elements = [element]
                current_page_number = element['metadata']['page_number']
            else:
                # Continue combining text for the current page
                current_combined_text += '\n' + element['text']
                current_page_elements.append(element)

        # Add the last page's results if any elements were processed
        if current_page_elements:
            combined_results.append({'content': current_combined_text, 'elements': current_page_elements})
        self.combined_elements = combined_results

    def image_interpretation(self):
        client = OpenAI(api_key='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')
        for i in range(len(self.filtered_elements)):
            for element in self.filtered_elements:
                if 'metadata' not in element:
                    continue
                if element['type'] == 'Image' and element['metadata']['image_mime_type'] in ['image/jpeg', 'image/png'] and element['metadata']['page_number']==28 :
                    page_number = element['metadata']['page_number']
                    no_image_elements = [element for element in self.combined_elements[page_number]['elements'] if
                                         element['type'] != 'Image']
                    completion = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"As Iris, the AI programming tutor integrated with Artemis on the "
                                                f"Technical University of Munich's platform, your assignment is to "
                                                f"dissect and elucidate an image from a university lecture slide in a "
                                                f"manner that aligns with academic standards. Your explanation should "
                                                f"thoroughly cover the technical or conceptual elements of the image, "
                                                f"framed within the lecture's overarching topic. You will receive "
                                                f"details on the slide's supplementary content, the placement of the "
                                                f"image, and its contextual background. Craft your explanation as a "
                                                f"university professor would—factual, direct, and devoid of personal "
                                                f"input or opinions, focusing solely on aiding student comprehension.\n"
                                                f"Context: \n {no_image_elements.__str__()}",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{element['metadata']['image_base64']}"
                                        }
                                    }
                                ],
                            }
                        ],
                        max_tokens=1000
                    )
                    for content_element in self.filtered_elements:
                        if 'page_number' not in content_element:
                            continue
                        if content_element['page_number'] == page_number:
                            content_element['content'] += f"\n{completion.choices[0].message.content}"

    def create_class(self):
        t2v = {
            "model": "ada",
            "modelVersion": "002",
            "type": "text",
            "vectorizeClassName": False
        }
        self.client.schema.delete_class("Lectures")
        if not self.client.schema.exists("Lectures"):
            class_obj = {
                "class": "Student_repositories",
                "vectorizer": "text2vec-openai",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"]
                    },
                    {
                        "name": "page_number",
                        "dataType": ["int"],
                    },
                    {
                        "name": "file_path",
                        "dataType": ["Text"],
                    },
                    {
                        "name": "stundent_ids",
                        "dataType": ["int"],
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
            self.batch_import()

    def batch_import(self):
        # Configure a batch process
        self.client.batch.configure(
            # `batch_size` takes an `int` value to enable auto-batching
            # dynamically update the `batch_size` based on import speed
            dynamic=True,
            timeout_retries=0
        )
        with self.client.batch as batch:
            # Batch import all Questions
            # Initialize the flag
            embeddings_created = False
            # create embeddings (exponential backoff to avoid RateLimitError)
            for j in range(5):  # max 5 retries
                # Only attempt to create embeddings if not already created
                if not embeddings_created:
                    try:
                        batch.add_data_object(
                            self.filtered_elements,
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


def generate_response(self, user_message, lecture_id):
    completion = query_openai(messages=[{
        "role": "user",
        "content": f"""
                                               Please give back lecture content that can answer this inquiry: 
                                               Do not add anything else.
                                               "{user_message}".\n
                                               """}])
    generated_lecture = completion.choices[0].message.content
    # add hypothetical document embeddings (hyde)
    response = (
        self.client.query
        .get("Lectures", ["content"])
        .with_near_text({"concepts": generated_lecture})
        # w.with_additional(f'rerank( query: "{user_message}", property: "content"){{score}}')
        .with_generate(grouped_task=prompt(user_message))
        .with_limit(3)
        .do()
    )
    generated_response = response["data"]["Get"]["Lectures"][0]["_additional"]["generate"]["groupedResult"]
    print(json.dumps(response, indent=2))
    return generated_response
