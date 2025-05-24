"""
OCR工具模块 - 提供商品名称和价格提取功能
"""
import os
import re
import json
import pytesseract
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import config

# 路径常量
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR_DIR = PROJECT_ROOT / "extractor"
PUBLIC_DIR = PROJECT_ROOT / "public"

# 输出文件路径
ITEM_INFO_PATH = EXTRACTOR_DIR / "item_info.json"
PRICE_INFO_PATH = EXTRACTOR_DIR / "price_info.json"

def setup_tesseract() -> Tuple[bool, str]:
    """
    设置Tesseract OCR路径
    返回: (成功状态, 信息消息)
    """
    # 检查环境变量
    if 'TESSERACT_CMD' in os.environ:
        pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']
        return True, f"从环境变量使用Tesseract路径: {os.environ['TESSERACT_CMD']}"
    
    # 尝试常见路径（Windows系统）
    if os.name == 'nt':
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True, f"使用Tesseract路径: {path}"
    
    # 测试是否可用（可能已在PATH中）
    try:
        pytesseract.get_tesseract_version()
        return True, "Tesseract OCR 已配置可用"
    except Exception as e:
        return False, f"未找到Tesseract OCR: {e}"

def get_image_files() -> List[Path]:
    """
    获取public目录中的所有图片文件
    """
    if not PUBLIC_DIR.exists():
        PUBLIC_DIR.mkdir(exist_ok=True)
    
    return list(PUBLIC_DIR.glob("*.png")) + list(PUBLIC_DIR.glob("*.jpg")) + list(PUBLIC_DIR.glob("*.jpeg"))

def setup_llm_client(api_key: str, model: str, api_base_url: Optional[str] = None):
    """
    初始化LLM客户端
    """
    from openai import OpenAI
    
    if api_base_url:
        return OpenAI(api_key=api_key, base_url=api_base_url)
    else:
        return OpenAI(api_key=api_key)

def extract_text_from_image(image_path: Path) -> str:
    """
    使用OCR从图片中提取文本
    """
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang='eng')

def extract_name_info(text: str, client, model: str) -> List[Dict[str, str]]:
    """
    从文本中提取商品名称信息
    """
    response = client.chat.completions.create(
        model=model,
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
    从文本中提取价格信息
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
    保存数据到JSON文件
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False

async def process_ocr_name(ctx) -> bool:
    """
    处理商品名称OCR提取流程
    """
    # 检查依赖和配置
    tesseract_ok, msg = setup_tesseract()
    if not tesseract_ok:
        await ctx.error(msg)
        return False
    
    await ctx.info("Tesseract OCR 已配置")
    
    # 获取配置
    api_key = config.API_KEY
    model = config.MODEL or "gpt-3.5-turbo"
    api_base_url = getattr(config, "URL", None)
    
    if not api_key:
        await ctx.error("API_KEY 未在config.py中设置或环境变量中未找到")
        return False
    
    # 获取图片文件
    image_files = get_image_files()
    await ctx.info(f"找到 {len(image_files)} 个图片文件")
    
    if not image_files:
        await ctx.error("没有找到任何图片文件")
        return False
    
    # 初始化客户端
    try:
        client = setup_llm_client(api_key, model, api_base_url)
        await ctx.info(f"使用模型: {model}" + (f" 和自定义API URL: {api_base_url}" if api_base_url else ""))
    except Exception as e:
        await ctx.error(f"初始化LLM客户端失败: {e}")
        return False
    
    # 处理所有图片
    item_name_info = []
    
    for path in image_files:
        try:
            await ctx.info(f"\n处理图片: {path.name}")
            
            # OCR文本提取
            text = extract_text_from_image(path)
            await ctx.info(f"提取的文本长度: {len(text)}")
            
            # 处理名称信息
            await ctx.info("调用LLM分析商品名称...")
            name_info = extract_name_info(text, client, model)
            
            if name_info:
                item_name_info.extend(name_info)
                await ctx.info(f"提取了 {len(name_info)} 条名称信息")
            else:
                await ctx.warning(f"从 {path.name} 中未提取到名称信息")
                
        except Exception as e:
            import traceback
            await ctx.error(f"处理图片 {path.name} 失败: {e}")
            await ctx.error(traceback.format_exc())
            continue
    
    # 保存结果
    if item_name_info:
        if save_json_data(item_name_info, ITEM_INFO_PATH):
            await ctx.info(f"名称信息已保存到 {ITEM_INFO_PATH}")
            return True
        else:
            await ctx.error(f"保存名称信息失败")
            return False
    else:
        await ctx.warning("没有提取到任何名称信息")
        return False

async def process_ocr_price(ctx) -> bool:
    """
    处理商品价格OCR提取流程
    """
    # 检查依赖和配置
    tesseract_ok, msg = setup_tesseract()
    if not tesseract_ok:
        await ctx.error(msg)
        return False
    
    await ctx.info("Tesseract OCR 已配置")
    
    # 获取配置
    api_key = config.API_KEY
    model = config.MODEL or "gpt-3.5-turbo"
    api_base_url = getattr(config, "URL", None)
    
    if not api_key:
        await ctx.error("API_KEY 未在config.py中设置或环境变量中未找到")
        return False
    
    # 获取图片文件
    image_files = get_image_files()
    await ctx.info(f"找到 {len(image_files)} 个图片文件")
    
    if not image_files:
        await ctx.error("没有找到任何图片文件")
        return False
    
    # 初始化客户端
    try:
        client = setup_llm_client(api_key, model, api_base_url)
        await ctx.info(f"使用模型: {model}" + (f" 和自定义API URL: {api_base_url}" if api_base_url else ""))
    except Exception as e:
        await ctx.error(f"初始化LLM客户端失败: {e}")
        return False
    
    # 处理所有图片
    item_price_info = []
    
    for path in image_files:
        try:
            await ctx.info(f"\n处理图片: {path.name}")
            
            # OCR文本提取
            text = extract_text_from_image(path)
            await ctx.info(f"提取的文本长度: {len(text)}")
            
            # 处理价格信息
            await ctx.info("调用LLM分析价格信息...")
            price_info = extract_price_info(text, client, model)
            
            if price_info:
                item_price_info.extend(price_info)
                await ctx.info(f"提取了 {len(price_info)} 条价格信息")
            else:
                await ctx.warning(f"从 {path.name} 中未提取到价格信息")
                
        except Exception as e:
            import traceback
            await ctx.error(f"处理图片 {path.name} 失败: {e}")
            await ctx.error(traceback.format_exc())
            continue
    
    # 保存结果：仅保存到 item_info.json
    if item_price_info:
        try:
            # 如果item_info.json已存在，尝试读取并添加价格信息
            if ITEM_INFO_PATH.exists():
                try:
                    with open(ITEM_INFO_PATH, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            # 直接将价格信息添加到已有的商品信息中
                            existing_data.extend(item_price_info)
                            if save_json_data(existing_data, ITEM_INFO_PATH):
                                await ctx.info(f"价格信息已添加到 {ITEM_INFO_PATH}")
                                return True
                            else:
                                await ctx.error(f"更新 {ITEM_INFO_PATH} 失败")
                                return False
                except Exception as e:
                    await ctx.error(f"读取或更新 {ITEM_INFO_PATH} 失败: {e}")
                    # 如果读取失败，尝试直接写入新文件
            
            # 如果文件不存在或读取失败，直接保存价格信息
            if save_json_data(item_price_info, ITEM_INFO_PATH):
                await ctx.info(f"价格信息已保存到 {ITEM_INFO_PATH}")
                return True
            else:
                await ctx.error(f"保存价格信息失败")
                return False
        except Exception as e:
            await ctx.error(f"保存到 {ITEM_INFO_PATH} 过程中出错: {e}")
            return False
    else:
        await ctx.warning("没有提取到任何价格信息")
        return False 