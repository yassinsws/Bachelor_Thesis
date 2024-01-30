import time
import fitz  # PyMuPDF
import openai
import weaviate
from openai import OpenAI
from weaviate.gql.get import HybridFusion
from unstructured.cleaners.core import clean
from unstructured.partition.pdf import partition_pdf
import os
import base64
from collections import defaultdict

import pickle


class AI:
    def __init__(self):
        self.filtered_elements = None

    # Function to encode images
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def image_interpretation(self):
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
        filtered_elements = [element for element in raw_pdf_elements if
                             element['type'] not in ['Footer', 'Header', 'UncategorizedText']]
        self.filtered_elements = filtered_elements

    def generate_response(self, user_message):
        client = OpenAI(api_key='')
        for element in self.filtered_elements:
            if element['type'] == 'Image' and element['metadata']['page_number'] == 6:
                base64_image = element['metadata']['image_base64']
                completion = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": 'In this message you will find an image coded in base64, can you explain '
                                            'to me the image provided ?',
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ],
                        }
                    ],
                    max_tokens=500
                )
                generated_lecture = completion.choices[0].message.content
                print(generated_lecture + '\n')
                return generated_lecture
        return 'no base 64 coding found!'
