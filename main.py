import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

def generate():
    api_key = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a helpful assistant that extracts newspaper fields from images.",
        generation_config={"response_mime_type": "application/json"}
    )

    image_path = "another-sample.jpg"
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "array", "items": {"type": "string"}},
            "subHeadline": {"type": "array", "items": {"type": "string"}},
            "byline": {"type": "array", "items": {"type": "string"}},
            "body": {"type": "array", "items": {"type": "string"}},
            "quotes": {"type": "array", "items": {"type": "string"}},
            "factboxes": {"type": "array", "items": {"type": "string"}}
        }
    }

    prompt = (
        "This is a newspaper image. "
        "Extract any of the following fields if available: "
        "headline, subHeadline, byline, body, quotes, factboxes. "
        "Return the result strictly matching this JSON schema:\n\n"
        f"{json.dumps(schema, indent=2)}"
    )

    response = model.generate_content([
        {"text": prompt},
        {"mime_type": "image/png", "data": image_bytes}
    ])

    raw_json = response.text
    data = json.loads(raw_json)
    print(data)

if __name__ == "__main__":
    generate()