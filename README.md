# Project Name

## Description

This project is a web application that processes food images to generate detailed nutritional information in JSON format. The application uses FastAPI to handle requests and OpenAI API for generating food details from the provided image URL.

## Features

- Upload an image or pass an image URL to get nutritional details.
- Uses OpenAI API to analyze and generate detailed food information in JSON format.
  
## Installation

Follow the steps below to set up and run the project.

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- pip (Python package manager)
- Node.js (if applicable)

### Steps

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/yourprojectname.git
    cd imagecaptionAPI
    ```

2. Install the required Python packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables (e.g., OpenAI API Key):
    - Create a `.env` file in the root directory and add your API key:
    ```env
    OPENAI_API_KEY=your_api_key_here
    ```

### Running the Application
    ```bash
    cd prototpe1(image base65)
    
    cd prototype2(image url)
    ```
1. Start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```
    open live stream server
livestrem ser
2. Navigate to `http://127.0.0.1:8000` in your browser to access the API or `http://127.0.0.1:8000/docs` for auto-generated Swagger documentation.

3. You can also test the `/process-url/` endpoint by sending a POST request with an image URL as a JSON payload.

### Example Request

To process an image, send a POST request to `http://127.0.0.1:8000/process-url/` with the following JSON body:

```json
{
    "image_url": "https://example.com/your-image.jpg"
}
