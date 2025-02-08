from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import re
import base64
from PIL import Image
from io import BytesIO
import logging

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Pydantic model for nutrition info
class NutritionInfo(BaseModel):
    food_name: str
    calorie: float
    protein: float
    carbs: float
    fat: float

# Image resizing function
def resize_image(image_data, output_path, scale=0.2):
    try:
        with Image.open(BytesIO(image_data)) as img:
            new_size = (int(img.width * scale), int(img.height * scale))
            resized_img = img.resize(new_size, Image.LANCZOS)
            resized_img.save(output_path)
        logging.debug("Image resized successfully")
    except Exception as e:
        logging.error(f"Image resize error: {e}")
        raise

@app.post("/image-caption")
async def process_image(file: UploadFile = File(...)):
    try:
        # Verify the file is an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Invalid file type. Please upload an image.")

        # Read the image file
        contents = await file.read()

        # Resize the image
        resized_image_path = "resized_image.jpg"
        resize_image(contents, resized_image_path, scale=0.2)

        # Encode the resized image
        with open(resized_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Clean up the resized image file
        os.remove(resized_image_path)

        # Prepare the data URL for OpenAI
        data_url = f"data:{file.content_type};base64,{base64_image}"

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
                        Generate the following details for a food item in JSON format:
                        food_name: The name of the dish in Thai.
                        calorie: Calorie count as number.
                        protein: Protein in grams as number.
                        carbs: Carbohydrates in grams as number.                          
                        fat: Fat in grams as number.
                        
                        Example:
                        {
                          "food_name": "ผัดไทย",
                          "calorie": 600,
                          "protein": 20,
                          "carbs": 75,
                          "fat": 25
                        }
                        """},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
        )

        # Log the raw response
        raw = response.choices[0].message.content
        logging.debug(f"Raw OpenAI Response: {raw}")

        # Clean and parse the response
        cleaned = re.sub(r'^```(json)?\s*|\s*```$', '', raw, flags=re.IGNORECASE)
        logging.debug(f"Cleaned Response: {cleaned}")

        result_json = json.loads(cleaned)
        validated = NutritionInfo(**result_json)
        return validated.dict()

    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}")
        raise HTTPException(422, f"Invalid JSON response: {str(e)}")
    except ValidationError as e:
        logging.error(f"Validation Error: {e}")
        raise HTTPException(422, f"Invalid data format: {str(e)}")
    except Exception as e:
        logging.error(f"Server Error: {e}")
        raise HTTPException(500, f"Server error: {str(e)}")