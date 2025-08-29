import json
import requests
import cairosvg
from openai import OpenAI

base_url_generation = "your_base_url_generation"
key_generation = "your_key_generation"

base_url_eval = "your_base_url_eval" # gemini
key_eval = "your_key_eval"

client = OpenAI(
    api_key=key_generation, 
    base_url=base_url_generation
)

def api_call_4_generation(prompt, model="o4-mini", max_tokens=10000):
    """
    Call the OpenAI API with the given prompt.
    
    Args:
        prompt (str): The prompt to send to the API
        model (str): The model to use for generation
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness (0-1)
        
    Returns:
        str: The generated response text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error: {str(e)}"

import base64
import io
import os
import tempfile
import re
from PIL import Image

def svg2img(svg_str, output_path, model):
    """
    Convert SVG code to a PNG image and save it to the specified path.
    If cairosvg is not available, returns None.

    Args:
        svg_str (str): SVG code as a string.
        output_path (str): Path to save the output PNG image.

    Returns:
        str: The output path if successful, None otherwise.
    """
    dir_name = "svg_img/" + model
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    output_path = dir_name + "/" + output_path
    
    try:
        cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), write_to=output_path)
        return output_path
    except Exception as e:
        print(f"Error converting SVG to image: {e}")
        return None

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        str: Base64 encoded string of the image, or None if an error occurs.
    """
    try:
        img = Image.open(image_path)
        # Handle transparency by converting transparent areas to white
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                bg.paste(img, mask=img.split()[3])
            elif img.mode == 'LA':
                bg.paste(img, mask=img.split()[1])
            elif img.mode == 'P' and 'transparency' in img.info:
                img_rgba = img.convert('RGBA')
                bg.paste(img_rgba, mask=img_rgba.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

def api_call_4_eval(model, svg_code, max_tokens=2000, output_path="output.png", prompt="Describe the image"):
    """
    Convert SVG code to a PNG image, encode it to base64, and call the API with the image and prompt.
    If cairosvg is not available, returns an error message.

    Args:
        model (str): Model name.
        svg_code (str): SVG code as a string.
        max_tokens (int): Maximum number of tokens for the API.
        output_path (str): Path to save the output PNG image.
        prompt (str): Prompt to send to the API.

    Returns:
        dict: A dictionary containing the API response or error message.
    """
    
    # Convert SVG to image
    img_path = svg2img(svg_code, output_path, model)
    if img_path is None:
        return {"success": False, "error": "failed to convert SVG to image"}

    # Encode image to base64
    image_base64 = encode_image_to_base64(img_path)
    if image_base64 is None:
        return {"success": False, "error": "failed to encode image to base64"}

    # Prepare API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key_eval}"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(base_url_eval, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        return {
            "success": True,
            "response": response_text
        }
    except Exception as e:
        print(f"Error calling API: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def load_json_file(filepath):
    """Load a JSON file and return its content"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_file(data, filepath):
    """Save data to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def test_api_img():
    """
    随机抽取3个entry，打印它们的id和description，以及API返回结果，用于测试代码正确性。
    """
    import random

    model = "gpt-4o-mini"
    generated_svg_path = "svg_generation_results/gpt-4o-mini_all_result.json"
    bench_path = "prompts/svg-compbench-prompts.json"

    with open(generated_svg_path, 'r') as f:
        generated_svg_data = json.load(f)
    with open(bench_path, 'r') as f:
        bench_data = json.load(f)

    # 随机抽样3个entry
    sample_n = min(3, len(generated_svg_data))
    sampled_items = random.sample(generated_svg_data, sample_n)

    for item in sampled_items:
        entry_id = item.get('id')
        # 尝试获取description字段，如果没有则用空字符串
        # Retrieve the description from bench_data by matching id
        prompt = ''
        entry_id = item.get('id')
        for bench_item in bench_data:
            if bench_item.get('id') == entry_id:
                prompt = bench_item.get('prompt', '')
                break
        svg_code = item.get('svg', '')
        output_path = f"output_{entry_id}.png"

        # 调用API
        query = f"check wether this img mathch the prompt: {prompt}"
        print(f"ID: {entry_id}")
        print(f"prompt: {prompt}")
        result = api_call_4_eval(model, svg_code, output_path=output_path, prompt=query)
        if result.get('success'):
            print(f"API Response: {result.get('response')}")
        else:
            print("Failed to get response from API.")
        print("-" * 40)

def check_svg_validity(svg_code):
    """Check if the SVG is valid XML by attempting to parse it with Python's XML libraries"""
    try:
        import xml.etree.ElementTree as ET
        try:
            ET.fromstring(svg_code)
            return True, None
        except ET.ParseError as e:
            error_msg = f"XML parsing error: {str(e)}"
            return False, error_msg
    except ImportError:
        # If ElementTree is not available, fall back to basic checks
        if not svg_code or not isinstance(svg_code, str):
            return False, "Empty or invalid SVG code"
        
        # Check for basic SVG structure
        if not svg_code.strip().startswith('<svg'):
            return False, "Missing SVG opening tag"
        
        if '</svg>' not in svg_code:
            return False, "Missing SVG closing tag"
        
        return True, None

if __name__ == "__main__":
    test_api_img()