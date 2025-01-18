from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# Define a request model
class ImageURLRequest(BaseModel):
    image_url: str

# Define your endpoint
@app.post("/process-url/")
async def process_url(request: ImageURLRequest):
    image_url = request.image_url

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            "Generate the following details for a food item in JSON format:
                            Menu: The name of the dish in (Thai Language).
                            Calorie: The calorie count of the dish.
                            Carbs: The amount of carbohydrates in grams.
                            Protein: The amount of protein in grams.
                            Fat: The amount of fat in grams."
                            #note no extra word!
                            example:
                            {
                            "Menu": "",
                            "Calorie": "",
                            "Carbs": "",
                            "Protein": "",
                            "Fat": ""
                            }
                            """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
        )
        raw = response.choices[0].message.content
        result = raw.strip("```json").strip().strip("```")
        result_json = json.loads(result)

        return {"result": result_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
