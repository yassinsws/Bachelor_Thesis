import os
import requests
import json
import fitz  # PyMuPDF
import lcamelot  # For tables
from openai.lib.azure import AzureOpenAI

api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("DEPLOYMENT_NAME")
API_KEY = os.getenv("AZURE_OPENAI_KEY")
azure_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15"
)

def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        for img in doc.get_page_images(page_num):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{output_folder}/image{page_num+1}_{xref}.{image_ext}"
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
    doc.close()

def extract_tables_from_pdf(pdf_path, output_folder):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
    for i, table in enumerate(tables):
        table.to_csv(f'{output_folder}/table_{i}.csv')


if __name__ == "__main__":

    base_url = f"{api_base}openai/deployments/{deployment_name}"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    # Prepare endpoint, headers, and request body
    endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview"
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": "Describe this picture:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.researchgate.net/profile/Yann-Gael-Gueheneuc/publication/249885094/figure"
                               "/fig27/AS:532128966377478@1503880840742/UML-class-diagram-for-Strategy-pattern.png"
                    }
                }
            ]}
        ],
        "max_tokens": 2000
    }

    # Make the API call
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    print(f"Status Code: {response.status_code}")
    print(response.text)
    #pdf_file = 'path/to/your/pdf.pdf'
    #output_dir = 'path/to/output/directory'
    #extract_images_from_pdf(pdf_file, output_dir)
    #extract_tables_from_pdf(pdf_file, output_dir)

