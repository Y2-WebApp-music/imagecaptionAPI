import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# Load the .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to count tokens
def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(tokenizer.encode(text))

# Path to your image
image_path = "image/krapaomoo-sub.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)
extra_condition = "พิเศษ"

# Constructing the message content
system_message_content = """
Generate the following details for a food item in JSON format:
Menu: The name of the dish in (Thai Language).
Calorie: The calorie count of the dish.
Carbs: The amount of carbohydrates in grams.
Protein: The amount of protein in grams.
Fat: The amount of fat in grams.
#note no extra word!

example1:
{
    "Menu": "",
    "Calorie": "",
    "Carbs": "",
    "Protein": "",
    "Fat": ""
}
"""

user_message_content = f"""
Here is an image of the dish along with an extra condition: {extra_condition}.
![Image](data:image/jpeg;base64,{base64_image})
"""
# Check token count for both messages separately
system_tokens = count_tokens(system_message_content)
user_tokens = count_tokens(user_message_content)

# Print the token counts
print(f"System message tokens: {system_tokens}")
print(f"User message tokens: {user_tokens}")

# Check token count
total_tokens = count_tokens(system_message_content + user_message_content)

if total_tokens > 3000:  # Adjust this limit based on the model's constraints
    raise ValueError("The input content exceeds the token limit.")

# API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": user_message_content},
    ],
    max_tokens=1000  # Adjust output token limit
)

# Extracting and printing the response
x = response.choices[0].message.content
print(x)
