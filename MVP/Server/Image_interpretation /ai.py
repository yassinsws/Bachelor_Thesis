from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
import os
import pickle


class AI:
    def __init__(self):
        self.filtered_elements = None

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
        self.filtered_elements = [element for element in raw_pdf_elements if
                                  element['type'] not in ['Footer', 'Header', 'UncategorizedText']]
        print(self.filtered_elements.__str__() + '\n')

    def generate_response(self, user_message):
        client = OpenAI(api_key='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')
        elements = [element for element in self.filtered_elements if element['type'] != 'Image' and element['metadata']['page_number'] == 28]
        base64_image = []
        for element in self.filtered_elements:
            if element['metadata']['page_number'] == 28 and element['type'] == 'Image':
                base64_image.append(element['metadata']['image_base64'])
        print(elements.__str__())
        completion = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'{user_message}\n'
                                    f'here is the context: {elements.__str__()}\n',
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image[0]}"
                            }
                        }
                    ],
                }
            ],
            max_tokens=1000
        )
        generated_lecture = completion.choices[0].message.content
        print(generated_lecture + '\n')
        if generated_lecture:
            return generated_lecture
        return 'no base 64 coding found!'
