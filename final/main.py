import os
import base64
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI(title="Nutrition Analyzer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def resize_and_encode_image(image_data: bytes, width: int = 586, height: int = 780) -> str:
    """Process uploaded image and return base64 string."""
    try:
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

SYSTEM_PROMPT = """Generate nutritional estimates for this Thai dish in JSON format. Follow these rules:
1. Menu: Thai name only (no English)
2. Values must be numerical approximations:
   - Calorie: Total: kcal 
   - Carbs/Protein/Fat: Grams 
3. Base estimates on common ingredients
4. Even uncertain, provide logical estimates
5. Strict JSON format (no markdown, no comments)

Example valid response food:
{
    "Menu": "กระเพราหมูสับไข่ดาว",
    "Calorie": 700,
    "Carbs": 45,
    "Protein": 35,
    "Fat": 40
}
"""

@app.post("/analyze", response_model=Dict[str, Any], summary="Analyze food nutrition from image")
async def analyze_nutrition(
    image: UploadFile = File(..., description="Food image (JPEG/PNG) under 5MB"),
    portion_size: str = Form("regular", description="Portion size specification")
):
    """
    Analyze uploaded food image and return nutrition estimates in JSON format.
    
    Returns:
        JSON object containing menu name and nutritional information
    """
    try:
        # Verify image type
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "Only JPEG/PNG images allowed")

        # Process image
        image_data = await image.read()
        base64_image = resize_and_encode_image(image_data)

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": f"Portion size: {portion_size}"}
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

@app.get("/test")
async def test_endpoint():
    """Health check endpoint"""
    return {"status": "active", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)