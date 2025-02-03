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

SYSTEM_PROMPT = """Generate nutritional estimates for this Thai dish in JSON format. Follow these rules:

1. food_name: Thai name only (no English)
2. Values must be numerical approximations:
   - calorie: Total kcal (300-1200 range)
   - protein/carbs/fat: Grams (10-100 range)
3. Base estimates on common Thai ingredients
4. Even uncertain, provide logical estimates
5. Strict JSON format (no markdown, no comments)

Example valid response:
{
    "food_name": "กระเพราหมูสับไข่ดาว",
    "calorie": 700,
    "protein": 35,
    "carbs": 45,
    "fat": 40
}
"""

class ImageRequest(BaseModel):
    image: str  # Must be raw base64 (without prefix)
    portion: str = "regular"

# Function to validate & process base64 image
def decode_and_resize_base64(image: str, width: int = 586, height: int = 780) -> str:
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
    
    Returns:
        JSON object containing food name and nutritional values.
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
            model="gpt-4o-mini",
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
