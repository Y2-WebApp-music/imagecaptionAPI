import os
import base64
import io
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from typing import Dict, Any
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI(title="Nutrition Analyzer API")

SYSTEM_PROMPT = """Analyze food images and provide an estimated nutritional breakdown in strict JSON format. Follow these rules carefully:
1. Use the key "food_name" to represent the dish name, and always provide the name in Thai only (no English translation).
2. Provide estimated numerical values for the following nutritional components:
   - "calorie": Total calories (kcal)
   - "protein": Protein content (grams)
   - "carbs": Carbohydrate content (grams)
   - "fat": Fat content (grams)
3. Base your estimations on common ingredients typically used in the dish.
4. If exact values are uncertain, provide reasonable and logical approximations.
5. The response must be in strict JSON format, without any markdown formatting, additional commentary, or explanations.
6. If the input is not a food item, return the following JSON response:
   {
       "food_name": "นี่ไม่ใช่อาหาร",
       "calorie": 0,
       "protein": 0,
       "carbs": 0,
       "fat": 0
   }
  
Example valid response:
{
    "food_name": "ต้มยำกุ้ง",
    "calorie": 350,
    "protein": 25,
    "carbs": 15,
    "fat": 20
}
"""

class ImageRequest(BaseModel):
    image: str  # Must be raw base64 (without prefix)
    portion: str = "regular"

# Function to validate & process base64 image
def decode_and_resize_base64(image: str, width: int = 250, height: int = 250) -> str:
    """Validates, decodes, and resizes base64 image."""
    try:
        # Decode the raw base64 image string (already stripped of metadata)
        image_data = base64.b64decode(image)

        with Image.open(io.BytesIO(image_data)) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            resized_img = img.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise HTTPException(400, f"Image processing failed: {str(e)}")

@app.post("/image-caption", response_model=Dict[str, Any], summary="Analyze food nutrition from base64 image")
async def analyze_nutrition(request: ImageRequest):
    """
    Analyze food image sent as a base64 string (without metadata) and return nutrition estimates.
    """
    try:
        # Validate & resize base64 image
        processed_image = decode_and_resize_base64(request.image)

        # Prepare request for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{processed_image}"}},
                {"type": "text", "text": f"Portion size: {request.portion}"}
            ]}
        ]

        # Get OpenAI response
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"},
            timeout=10  # Increased timeout for image processing
        )

        return eval(response.choices[0].message.content)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
