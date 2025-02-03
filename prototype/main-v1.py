from fastapi import FastAPI, HTTPException
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

# Define a request model
class ImageURLRequest(BaseModel):
    image_url: str

# Define your endpoint
@app.post("/image-caption")
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
                            food_name: The name of the dish in (Thai Language).
                            calorie: The calorie count of the dish.
                            protein: The amount of protein in grams.
                            carbs: The amount of carbohydrates in grams.                          
                            fat: The amount of fat in grams."
                            #note no extra word! and value must be number except food_name
                            example:
                            {
                            "food_name": "",
                            "calorie": "",
                            "protein": "",
                            "carbs": "",                           
                            "fat": ""
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

        return result_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))