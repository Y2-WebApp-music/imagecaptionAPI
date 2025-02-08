import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from PIL import Image  # Import Pillow for image processing
import json  

# Load the .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Function to count tokens
def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    return len(tokenizer.encode(text))


# Resize image to 20% of its original size
def resize_image(image_path, output_path, scale=0.2):
    with Image.open(image_path) as img:
        new_size = (int(img.width * scale), int(img.height * scale))
        resized_img = img.resize(new_size, Image.LANCZOS)
        resized_img.save(output_path)

# Local image path 
image_path = r"image\krapaomoo-sub.jpg"  # Added raw string for Windows path
resized_image_path = "image/kangjead_resized.jpeg"

# Resize and encode image once (shared for both sizes)
resize_image(image_path, resized_image_path, scale=0.20)
with open(resized_image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Comparison function
# Updated comparison function
def compare_results(regular_data, large_data):
    print("\n=== PORTION COMPARISON ===")
    
    key_mapping = {
        "menu": "Menu",
        "calorie": "Calorie",
        "carbs": "Carbs",
        "protein": "Protein",
        "fat": "Fat"
    }
    
    for raw_key in ["menu", "calorie", "carbs", "protein", "fat"]:
        key = key_mapping.get(raw_key, raw_key)
        try:
            r = float(regular_data.get(key, regular_data.get(raw_key, 0)))
            l = float(large_data.get(key, large_data.get(raw_key, 0)))
            print(f"{key}: {r} → {l} ({l-r:+g})")
        except:
            print(f"{key}: Comparison failed")

    # Name check remains same
    menu_match = regular_data.get("Menu", "") == large_data.get("Menu", "")
    print(f"\nMenu: {'MATCH' if menu_match else 'DIFFERENT'}")

    # Name consistency check
    menu_match = regular_data.get("Menu", "") == large_data.get("Menu", "")
    print(f"\nMenu: {'MATCH' if menu_match else 'DIFFERENT'}")
    if not menu_match:
        print(f"  Regular: {regular_data.get('Menu', '')}")
        print(f"  Large: {large_data.get('Menu', '')}")
# Test both portion sizes
results = {}
for portion in ["regular", "large"]:
    print(f"\n=== PROCESSING {portion.upper()} PORTION ===")
    
    # Modified system prompt with portion awareness
    system_message_content = f"""Generate nutritional estimates for this Thai dish in JSON format. Rules:
1. Menu: Thai name only
2. Values as numbers for {portion} portion:
   - Calorie: {(300,500) if portion == 'regular' else (500,1000)} kcal range
   - Carbs/Protein/Fat: {(15,60) if portion == 'regular' else (30,100)}g range
3. Base on common ingredients
4. Strict JSON format
5. json key english only"""
    

    user_message_content = [
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",  # Fixed MIME type
                "detail": "auto"
            }
        },
        {
            "type": "text", 
            "text": f"Analyze this {portion} portion dish."
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content},
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        results[portion] = json.loads(result)  # Safer parsing
        print(f"{portion.upper()} RESULT:")
        print(json.dumps(results[portion], indent=2, ensure_ascii=False))
        
    except Exception as e:
         print(f"Error processing {portion}: {str(e)}")
         results[portion] = None

# Perform comparison if both results exist
if results["regular"] and results["large"]:
    compare_results(results["regular"], results["large"])
else:
    print("\nComparison not possible due to missing data")

# Cleanup
if os.path.exists(resized_image_path):
    os.remove(resized_image_path)