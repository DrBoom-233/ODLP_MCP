import os
import re
import pytesseract
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import json
import config
from pathlib import Path


# Initialize OpenAI client
client = OpenAI(
    api_key=config.API_KEY,
    # base_url=config.URL
)

# 拿到项目根 = extractor 脚本的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# public 目录的绝对路径
SCREENSHOT_DIR = PROJECT_ROOT / "public"

def get_screenshot_paths():
    # 确保目录存在
    if not SCREENSHOT_DIR.is_dir():
        raise FileNotFoundError(f"找不到截图目录：{SCREENSHOT_DIR}")

    return [
        str(SCREENSHOT_DIR / file)
        for file in SCREENSHOT_DIR.iterdir()
        if file.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]


# Main execution flow
screenshot_paths = get_screenshot_paths()
item_info = []

# Process each screenshot
for path in screenshot_paths:
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang='eng')
    print(text)


    # Use OpenAI API to analyze extracted text
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": "You are an information extraction assistant."},
            {"role": "user", "content": text},
            {"role": "system",
             "content": "Summarize the item information from the text above. "
                        "List all item names and its orders in format. "
                        "Do not put any price info into json, that's not your job!"
                        "json format should be like{order:, item: }"}
        ]
    )

    # Extract GPT response
    gpt_response = response.choices[0].message.content
    print(gpt_response)

    # Clean response
    cleaned_response = re.sub(r'```json|```', '', gpt_response).strip()

    # Try to load JSON data
    try:
        extracted_info = json.loads(cleaned_response)
        if isinstance(extracted_info, list):
            item_info.extend(extracted_info)
        else:
            print(f"Expected a list but got {type(extracted_info)} for {path}.")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON for {path}: {cleaned_response}")
        continue

# Save item_info to item_info.json
if item_info:
    with open('item_info.json', 'w', encoding='utf-8') as json_file:
        json.dump(item_info, json_file, ensure_ascii=False, indent=4)
    print("All screenshots have been processed and the item information has been saved to item_info.json.")
else:
    print("No valid item information extracted from screenshots.")