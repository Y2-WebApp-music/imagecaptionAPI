import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from PIL import Image  # Import Pillow for image processing

# Load the .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to count tokens
def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))

# Function to extract text content from message
def get_text_content(message_content):
    if isinstance(message_content, str):
        return message_content
    elif isinstance(message_content, list):
        text_parts = []
        for part in message_content:
            if part.get("type") == "text" and "text" in part:
                text_parts.append(part["text"])
        return " ".join(text_parts)
    else:
        return ""

# Resize image to 20% of its original size
def resize_image(image_path, output_path, scale=0.2):
    with Image.open(image_path) as img:
        new_size = (int(img.width * scale), int(img.height * scale))
        resized_img = img.resize(new_size, Image.LANCZOS)
        resized_img.save(output_path)

# Local image path (replace with your actual image path)
image_path ="image\kangjead.jpeg"
resized_image_path = "image/kangjead_resized.jpeg"
portion_sizes = "regular"

# Resize the image before encoding
resize_image(image_path, resized_image_path, scale=0.20)

# Read and encode resized image
with open(resized_image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Constructing the message content
system_message_content = """Generate nutritional estimates for this Thai dish in JSON format. Follow these rules:

1. Menu: Thai name only (no English)
2. Values must be numerical approximations:
   - Calorie: Total kcal (300-1200 range)
   - Carbs/Protein/Fat: Grams (10-100 range)
3. Base estimates on common Thai ingredients
4. Even uncertain, provide logical estimates
5. Strict JSON format (no markdown, no comments)

Example valid response:
{
    "Menu": "กระเพราหมูสับไข่ดาว",
    "Calorie": 700,
    "Carbs": 45,
    "Protein": 35,
    "Fat": 40
}
"""

user_message_content = [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
    {"type": "text", "text": f"Here is an image of the dish along with an portion sizes: {portion_sizes}"},
]

# Check token count for text portions
system_text = get_text_content(system_message_content)
user_text = get_text_content(user_message_content)

system_tokens = count_tokens(system_text)
user_tokens = count_tokens(user_text)

# Print the token counts
print(f"System message tokens: {system_tokens}")
print(f"User message tokens: {user_tokens}")
print(f"portion_sizes: {portion_sizes}")

# Check token count (note: this doesn't include image tokens)
total_tokens = system_tokens + user_tokens

if total_tokens > 3000:  # Adjust this limit based on the model's constraints
    raise ValueError("The input content exceeds the token limit.")

# API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": user_message_content},
    ],
     # Added enhancements
    temperature=0.5,  # For consistent formatting
    response_format={"type": "json_object"}  # Enforce JSON mode
)

# Extracting and printing the response
x = response.choices[0].message.content
print(x)
print(len(base64_image)+total_tokens)
