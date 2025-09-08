"""
OCR Tool Module - Provides functionality for extracting product names and prices
"""
import os
import re
import json
import pytesseract
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import config

# Path constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR_DIR = PROJECT_ROOT / "extractor"
PUBLIC_DIR = PROJECT_ROOT / "public"

# Output file paths
ITEM_INFO_PATH = EXTRACTOR_DIR / "item_info.json"
PRICE_INFO_PATH = EXTRACTOR_DIR / "price_info.json"

def setup_tesseract() -> Tuple[bool, str]:
    """
    Set up Tesseract OCR path
    Returns: (Success status, Information message)
    """
    # Check environment variables
    if 'TESSERACT_CMD' in os.environ:
        pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']
        return True, f"Using Tesseract path from environment variable: {os.environ['TESSERACT_CMD']}"
    
    # Try common paths (Windows systems)
    if os.name == 'nt':
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True, f"Using Tesseract path: {path}"
    
    # Test if available (might already be in PATH)
    try:
        pytesseract.get_tesseract_version()
        return True, "Tesseract OCR is configured and available"
    except Exception as e:
        return False, f"Tesseract OCR not found: {e}"

def get_image_files() -> List[Path]:
    """
    Get all image files in the public directory
    """
    if not PUBLIC_DIR.exists():
        PUBLIC_DIR.mkdir(exist_ok=True)
    
    return list(PUBLIC_DIR.glob("*.png")) + list(PUBLIC_DIR.glob("*.jpg")) + list(PUBLIC_DIR.glob("*.jpeg"))

def setup_llm_client(api_key: str, model: str, api_base_url: Optional[str] = None, service_type: str = "openai"):
    """
    Initialize the LLM client
    
    Args:
        api_key: API key
        model: Model name
        api_base_url: API base URL (optional)
        service_type: Service type, "openai" or "deepseek"
    """
    from openai import OpenAI
    
    if service_type.lower() == "openai":
        # Use OpenAI's API, ignore custom base_url
        return OpenAI(api_key=api_key)
    elif service_type.lower() == "deepseek":
        # Use DeepSeek's API, requires custom base_url
        if not api_base_url:
            raise ValueError("Using DeepSeek service requires providing an API base URL")
        return OpenAI(api_key=api_key, base_url=api_base_url)
    else:
        # Default behavior remains unchanged
        if api_base_url:
            return OpenAI(api_key=api_key, base_url=api_base_url)
        else:
            return OpenAI(api_key=api_key)

def extract_text_from_image(image_path: Path) -> str:
    """
    Extract text from an image using OCR
    """
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang='eng')

def extract_name_info(text: str, client, model: str) -> List[Dict[str, str]]:
    """
    Extract product name information from text
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an information extraction assistant."},
            {"role": "user", "content": text},
            {"role": "system",
             "content": "Summarize the item information from the text above. "
                        "List all item names and its orders in format. "
                        "In some cases, you might receive some information about the housing rental information,"
                        "you should list the treat the address information as item name."
                        "Do not put any price info into json, that's not your job!"
                        "json format should be like{order:, item: }"}
        ]
    )
    
    name_response = response.choices[0].message.content
    cleaned_response = re.sub(r'```json|```', '', name_response).strip()
    
    try:
        extracted_info = json.loads(cleaned_response)
        if isinstance(extracted_info, list):
            return extracted_info
        elif isinstance(extracted_info, dict):
            return [extracted_info]
        else:
            return []
    except json.JSONDecodeError:
        return []

def extract_price_info(text: str, client, model: str) -> List[Dict[str, str]]:
    """
    Extract price information from text
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an information extraction assistant."},
            {"role": "user", "content": text},
            {"role": "system",
             "content": "Summarize the price information from the text above. "
                        "List all prices and its orders in format. "
                        "Do not put any product name info into json, that's not your job!"
                        "json format should be like{order:, price: }"}
        ]
    )
    
    price_response = response.choices[0].message.content
    cleaned_response = re.sub(r'```json|```', '', price_response).strip()
    
    try:
        extracted_info = json.loads(cleaned_response)
        if isinstance(extracted_info, list):
            return extracted_info
        elif isinstance(extracted_info, dict):
            return [extracted_info]
        else:
            return []
    except json.JSONDecodeError:
        return []

def save_json_data(data: List[Dict[str, Any]], file_path: Path) -> bool:
    """
    Save data to a JSON file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False

async def process_ocr_name(ctx, service_type: str = "openai") -> bool:
    """
    Process the OCR extraction workflow for product names
    
    Args:
        ctx: Context object
        service_type: Service type, "openai" or "deepseek"
    """
    # Check dependencies and configuration
    tesseract_ok, msg = setup_tesseract()
    if not tesseract_ok:
        await ctx.error(msg)
        return False
    
    await ctx.info("Tesseract OCR is configured")
    
    # Get configuration based on service type
    if service_type.lower() == "openai":
        api_key = config.OPENAI_API_KEY
        model = config.OPENAI_MODEL
        api_base_url = None
    elif service_type.lower() == "deepseek":
        api_key = config.API_KEY  # DeepSeek API key
        model = config.CHAT_MODEL  # DeepSeek chat model
        api_base_url = config.URL  # DeepSeek API URL
    else:
        # Default behavior remains unchanged
        api_key = config.OPENAI_API_KEY
        model = config.OPENAI_MODEL
        api_base_url = getattr(config, "URL", None)
    
    if not api_key:
        await ctx.error("API_KEY is not set in config.py or not found in environment variables")
        return False
    
    # Get image files
    image_files = get_image_files()
    await ctx.info(f"Found {len(image_files)} image files")
    
    if not image_files:
        await ctx.error("No image files found")
        return False
    
    # Initialize client
    try:
        client = setup_llm_client(api_key, model, api_base_url, service_type)
        await ctx.info(f"Using service: {service_type}, model: {model}" + (f" and custom API URL: {api_base_url}" if api_base_url else ""))
    except Exception as e:
        await ctx.error(f"Failed to initialize LLM client: {e}")
        return False
    
    # Process all images
    item_name_info = []
    
    for path in image_files:
        try:
            await ctx.info(f"\nProcessing image: {path.name}")
            
            # OCR text extraction
            text = extract_text_from_image(path)
            await ctx.info(f"Extracted text length: {len(text)}")
            
            # Process name information
            await ctx.info("Calling LLM to analyze product names...")
            name_info = extract_name_info(text, client, model)
            
            if name_info:
                item_name_info.extend(name_info)
                await ctx.info(f"Extracted {len(name_info)} name information entries")
            else:
                await ctx.warning(f"No name information extracted from {path.name}")
                
        except Exception as e:
            import traceback
            await ctx.error(f"Failed to process image {path.name}: {e}")
            await ctx.error(traceback.format_exc())
            continue
    
    # Save results
    if item_name_info:
        if save_json_data(item_name_info, ITEM_INFO_PATH):
            await ctx.info(f"Name information saved to {ITEM_INFO_PATH}")
            return True
        else:
            await ctx.error("Failed to save name information")
            return False
    else:
        await ctx.warning("No name information extracted")
        return False

async def process_ocr_price(ctx, service_type: str = "openai") -> bool:
    """
    Process the OCR extraction workflow for product prices
    
    Args:
        ctx: Context object
        service_type: Service type, "openai" or "deepseek"
    """
    # Check dependencies and configuration
    tesseract_ok, msg = setup_tesseract()
    if not tesseract_ok:
        await ctx.error(msg)
        return False
    
    await ctx.info("Tesseract OCR is configured")
    
    # Get configuration based on service type
    if service_type.lower() == "openai":
        api_key = config.OPENAI_API_KEY
        model = config.OPENAI_MODEL
        api_base_url = None
    elif service_type.lower() == "deepseek":
        api_key = config.API_KEY  # DeepSeek API key
        model = config.CHAT_MODEL  # DeepSeek chat model
        api_base_url = config.URL  # DeepSeek API URL
    else:
        # Default behavior remains unchanged
        api_key = config.OPENAI_API_KEY
        model = config.OPENAI_MODEL
        api_base_url = getattr(config, "URL", None)
    
    if not api_key:
        await ctx.error("API_KEY is not set in config.py or not found in environment variables")
        return False
    
    # Get image files
    image_files = get_image_files()
    await ctx.info(f"Found {len(image_files)} image files")
    
    if not image_files:
        await ctx.error("No image files found")
        return False
    
    # Initialize client
    try:
        client = setup_llm_client(api_key, model, api_base_url, service_type)
        await ctx.info(f"Using service: {service_type}, model: {model}" + (f" and custom API URL: {api_base_url}" if api_base_url else ""))
    except Exception as e:
        await ctx.error(f"Failed to initialize LLM client: {e}")
        return False
    
    # Process all images
    item_price_info = []
    
    for path in image_files:
        try:
            await ctx.info(f"\nProcessing image: {path.name}")
            
            # OCR text extraction
            text = extract_text_from_image(path)
            await ctx.info(f"Extracted text length: {len(text)}")
            
            # Process price information
            await ctx.info("Calling LLM to analyze price information...")
            price_info = extract_price_info(text, client, model)
            
            if price_info:
                item_price_info.extend(price_info)
                await ctx.info(f"Extracted {len(price_info)} price information entries")
            else:
                await ctx.warning(f"No price information extracted from {path.name}")
                
        except Exception as e:
            import traceback
            await ctx.error(f"Failed to process image {path.name}: {e}")
            await ctx.error(traceback.format_exc())
            continue
    
    # Save results: Only save to item_info.json
    if item_price_info:
        try:
            # If item_info.json exists, try to read and add price information
            if ITEM_INFO_PATH.exists():
                try:
                    with open(ITEM_INFO_PATH, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            # Directly add price information to existing product information
                            existing_data.extend(item_price_info)
                            if save_json_data(existing_data, ITEM_INFO_PATH):
                                await ctx.info(f"Price information added to {ITEM_INFO_PATH}")
                                return True
                            else:
                                await ctx.error(f"Failed to update {ITEM_INFO_PATH}")
                                return False
                except Exception as e:
                    await ctx.error(f"Failed to read or update {ITEM_INFO_PATH}: {e}")
                    # If reading fails, try to write directly to a new file
            
            # If the file does not exist or reading fails, directly save price information
            if save_json_data(item_price_info, ITEM_INFO_PATH):
                await ctx.info(f"Price information saved to {ITEM_INFO_PATH}")
                return True
            else:
                await ctx.error("Failed to save price information")
                return False
        except Exception as e:
            await ctx.error(f"Error occurred while saving to {ITEM_INFO_PATH}: {e}")
            return False
    else:
        await ctx.warning("No price information extracted")
        return False