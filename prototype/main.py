from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# Load the .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Define your endpoint
@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": """
                    "Generate the following details for a food item in JSON format:
                    Menu: The name of the dish in (Thai Language) .
                    Calorie: The calorie count of the dish.
                    Carbs: The amount of carbohydrates in grams.
                    Protein: The amount of protein in grams.
                    Fat: The amount of fat in grams."
                    #note no extra word!
                    example:
                    {
                    "Menu": "",
                    "Calorie": "",
                    "Carbs":"",
                    "Protein"":,
                    "Fat": ""
                    }
                    """,
                    },
                ],
            }
             {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": """

                    """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    raw = response.choices[0].message.content
    result = raw.strip("```json").strip().strip("```")
    result_json = json.loads(result)

    return {"result": result_json}
