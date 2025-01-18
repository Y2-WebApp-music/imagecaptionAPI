import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "image\krapaomoo-sub.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    #gpt-4o
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
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
                    example1:
                    {
                    "Menu": "",
                    "Calorie": "",
                    "Carbs":"",
                    "Protein"":,
                    "Fat": ""
                    }
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

x = (response.choices[0].message.content)
print(x)