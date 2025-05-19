import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import config

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=config.API_KEY,
    # base_url=config.URL
)

# Step 1: Read item-price data from BeautifulSoup_Content.json
with open('BeautifulSoup_Content.json', 'r', encoding='utf-8') as f:
    content_data = json.load(f)

# Prepare lists to store GPT responses and extracted price information
responses = []
price_info = []

# Process each item in content_data sorted by the 'Order' field
for item in sorted(content_data, key=lambda x: x['Order']):
    content_text = item['Content']

    # Use OpenAI API to analyze and extract text
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an information extraction assistant. You are assisting with CPI calculation."
            },
            {"role": "user", "content": content_text},
            {
                "role": "system",
                "content": (
                    "You are receiving text from a JSON file that contains the names of items, their corresponding price information, "
                    "and a data-product-code extracted from the HTML content. Some items may have multiple prices, each with a different unit of measure. "
                    "Please extract all the price information and the data-product-code for each item. "
                    "The output should be in JSON format with each object containing the fields \"item\", \"data_product_code\", and \"prices\", "
                    "where \"prices\" is an array of objects, each containing \"price\" and \"unit\" fields. "
                    "For example: {\"item\": \"cucumber\", \"data_product_code\": \"ABC123\", \"prices\": [{\"price\": 2.99, \"unit\": \"ea.\"}, {\"price\": 1.99, \"unit\": \"kg\"}]}. "
                    "If there is no relevant information, output {\"item\": null, \"data_product_code\": null, \"prices\": []}. "
                    "Make sure to output exactly one JSON object per item, and do not combine multiple items into one JSON."
                )
            }
        ]
    )

    # Extract GPT response
    gpt_response = response.choices[0].message.content
    responses.append(gpt_response)

    # Clean up the response to remove JSON formatting artifacts
    cleaned_response = re.sub(r'```json|```', '', gpt_response).strip()

    # Attempt to parse the JSON data
    try:
        extracted_info = json.loads(cleaned_response)
        # Check if the JSON has the expected structure: "item", "data_product_code", and "prices" fields
        if (isinstance(extracted_info, dict) and
                "item" in extracted_info and
                "data_product_code" in extracted_info and
                "prices" in extracted_info):
            price_info.append(extracted_info)
        else:
            print(f"Unexpected format for extracted data: {extracted_info}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON for response: {cleaned_response}")

# Save extracted price information to price_info.json file
if price_info:
    with open('price_info.json', 'w', encoding='utf-8') as json_file:
        json.dump(price_info, json_file, ensure_ascii=False, indent=4)
    print("All items have been processed and the price information has been saved to price_info.json.")
else:
    print("No valid price information extracted from the data.")

# Optionally, print all responses for review
for response in responses:
    print(response)


# Function to remove entries with null values from price_info.json
def clean_price_info(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Filter out entries where "item" is null or where "prices" is empty
    cleaned_data = [entry for entry in data if entry.get("item") is not None and entry.get("prices")]

    # Save cleaned data back to the file
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(cleaned_data, json_file, ensure_ascii=False, indent=4)
    print(f"Cleaned data saved back to {file_path}.")


# Clean price_info.json by removing entries with null or empty price arrays
clean_price_info('price_info.json')
