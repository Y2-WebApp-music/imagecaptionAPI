import os
import io
import base64
import openai
from dotenv import load_dotenv
import tiktoken
from PIL import Image

# Load environment variables
load_dotenv()

# Instead of setting the API key directly on openai, instantiate the new client:
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to count tokens using tiktoken
def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))

def resize_and_encode_image(image_data: bytes, width: int = 586, height: int = 780) -> str:
    """
    Resize the image in memory and return a base64 encoded JPEG string.
    """
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            resized_img = img.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")

# Read and process the image from disk
image_path = "final/image/tongkutsu.jpg"
with open(image_path, "rb") as image_file:
    original_image_data = image_file.read()

base64_image = resize_and_encode_image(original_image_data, width=480, height=480)
portion_sizes = "regular"
print("Length of base64 image string:", len(base64_image))

# Construct system and user messages
system_message_content = system_message_content = """
provide nutritional estimates following these rules:
1. Menu: Thai name only (no English)
2. Values must be numerical approximations:
   - Calorie: Total kcal 
   - Carbs/Protein/Fat: Grams 
3. Base estimates on common Thai ingredients.
4. Even if uncertain, provide logical estimates.
5. Use strict JSON format (no markdown, no comments).

Example valid response:
{
    "Menu": "กระเพราหมูสับไข่ดาว",
    "Calorie": 700,
    "Carbs": 45,
    "Protein": 35,
    "Fat": 40
}
"""


user_message_content = (
    f"Image (base64): data:image/jpeg;base64,{base64_image}"
    f"Note: Ensure that the image is of a Thai dish. Here is an image along with a portion size: {portion_sizes}"
)


# Calculate token counts for debugging or logging
system_tokens = count_tokens(system_message_content)
user_tokens = count_tokens(user_message_content)
total_tokens = system_tokens + user_tokens

# Use the new client interface to call the chat completions endpoint.
response = client.chat.completions.create(
    model="gpt-4o",  # or use "gpt-4" as appropriate
    messages=[
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": user_message_content},
    ], 
    temperature=0.5,  # for consistent output
    response_format={"type": "json_object"}
)

# Extract and print the response
ai_result = response.choices[0].message.content.strip()
print("Response from OpenAI API:")
print(ai_result)

# Print a summary of the base64 string length plus token count
print("Sum of base64 string length and total tokens:", len(base64_image) + total_tokens)
