import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import easyocr
import cv2
from rapidfuzz import fuzz

load_dotenv()

def generate(image_path):
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="You are a helpful assistant that extracts newspaper fields from images.",
        generation_config={"response_mime_type": "application/json"}
    )

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
    pretty = json.dumps(data, indent=2)
    print(pretty)

    return data, image_path

def should_merge(a_tl, a_br, b_tl, b_br, gap_x=20, gap_y=50):
    """
    Returns True if rectangles overlap or are within gap thresholds.
    """
    if a_br[0] + gap_x < b_tl[0]:  # a is strictly left of b
        return False
    if b_br[0] + gap_x < a_tl[0]:  # b is strictly left of a
        return False
    if a_br[1] + gap_y < b_tl[1]:  # a is strictly above b
        return False
    if b_br[1] + gap_y < a_tl[1]:  # b is strictly above a
        return False
    return True

def draw_boxes_from_data(data, image_path):

    # Define a color for each field
    colors = {
        "headline": (255, 0, 0),
        "subHeadline": (0, 255, 0),
        "byline": (0, 0, 255),
        "body": (255, 255, 0),
        "quotes": (255, 0, 255),
        "factboxes": (0, 255, 255),
    }

    reader = easyocr.Reader(['en'])
    image = cv2.imread(image_path)
    results = reader.readtext(image)

    validated = {"headline": []}
    headline_text = " ".join(data['headline'])

    for coordinates, text, _ in results:
        score = fuzz.partial_ratio(headline_text, text)

        if score > 80 and len(text.split()) > 1:
            print(text)
            # remove the matched text once from headline_text
            headline_text = headline_text.replace(text, "", 1).strip()

            tl = coordinates[0]  # top-left
            br = coordinates[2]  # bottom-right
            top_left = (int(tl[0]), int(tl[1]))
            bottom_right = (int(br[0]), int(br[1]))

            merged = False
            for i, (ex_tl, ex_br) in enumerate(validated["headline"]):
                if should_merge(ex_tl, ex_br, top_left, bottom_right):
                    new_tl = (min(ex_tl[0], top_left[0]), min(ex_tl[1], top_left[1]))
                    new_br = (max(ex_br[0], bottom_right[0]), max(ex_br[1], bottom_right[1]))
                    validated["headline"][i] = (new_tl, new_br)
                    merged = True
                    break

            if not merged:
                validated["headline"].append((top_left, bottom_right))

    # Draw merged bounding boxes
    for top_left, bottom_right in validated["headline"]:
        cv2.rectangle(image, top_left, bottom_right, colors["headline"], 2)

    print("done") 
    output_path = f"{os.path.splitext(image_path)[0]}_result{os.path.splitext(image_path)[1]}"
    cv2.imwrite(output_path, image)
    print(f"Saved boxed image to {output_path}")

if __name__ == "__main__":
    image_path = "sample3.png"
    data, image_path = generate(image_path)
    draw_boxes_from_data(data, image_path)