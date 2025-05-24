import os
import re
import sys
import pytesseract
from PIL import Image
from pathlib import Path
import json
import traceback

# 打印系统信息用于调试
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# 配置 Tesseract 路径
if os.name == 'nt':  # Windows
    # 尝试几个常见的 Tesseract 安装路径
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\Charlie\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"找到 Tesseract 路径: {path}")
            break
    
    # 从环境变量获取
    if 'TESSERACT_CMD' in os.environ:
        pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']
        print(f"从环境变量获取 Tesseract 路径: {os.environ['TESSERACT_CMD']}")

try:
    from openai import OpenAI
    print("成功导入 OpenAI 库")
except ImportError:
    print("错误: 未安装 OpenAI 库，请运行 'pip install openai'")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("成功导入 dotenv 库")
except ImportError:
    print("错误: 未安装 python-dotenv 库，请运行 'pip install python-dotenv'")
    sys.exit(1)

# 加载环境变量
load_dotenv()
print("已加载 .env 环境变量文件")

# 检查 config 文件
try:
    import config
    print("成功导入 config 模块")
except ImportError:
    print("错误: 找不到 config.py 文件，或者导入失败")
    sys.exit(1)

# 检查 API_KEY
if not hasattr(config, 'API_KEY') or not config.API_KEY:
    print("错误: config.py 中未定义 API_KEY 或为空")
    api_key_env = os.environ.get('OPENAI_API_KEY')
    if not api_key_env:
        print("错误: 环境变量 OPENAI_API_KEY 也未设置")
        sys.exit(1)
    else:
        print("使用环境变量中的 OPENAI_API_KEY")
        config.API_KEY = api_key_env

# 检查 MODEL
if not hasattr(config, 'MODEL') or not config.MODEL:
    print("错误: config.py 中未定义 MODEL 或为空")
    model_env = os.environ.get('OPENAI_MODEL')
    if not model_env:
        print("警告: 环境变量 OPENAI_MODEL 也未设置，使用默认值 'gpt-3.5-turbo'")
        config.MODEL = 'gpt-3.5-turbo'
    else:
        print(f"使用环境变量中的 OPENAI_MODEL: {model_env}")
        config.MODEL = model_env

print(f"使用的 API_KEY (前5位): {config.API_KEY[:5]}...")
print(f"使用的 MODEL: {config.MODEL}")

# 检查 Tesseract 是否安装
try:
    tesseract_version = pytesseract.get_tesseract_version()
    print(f"已安装 Tesseract OCR 版本: {tesseract_version}")
except Exception as e:
    print(f"错误: Tesseract OCR 未安装或配置错误: {e}")
    print("请安装 Tesseract OCR 并确保路径正确设置")
    if os.name == 'nt':  # Windows
        print("Windows 用户请从 https://github.com/UB-Mannheim/tesseract/wiki 下载并安装 Tesseract")
        print("然后设置环境变量 TESSERACT_CMD 或在代码中设置路径:")
        print("pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
    sys.exit(1)

# Initialize OpenAI client
try:
    client = OpenAI(
        api_key=config.API_KEY,
        # base_url=config.URL
    )
    print("成功初始化 OpenAI 客户端")
except Exception as e:
    print(f"错误: 初始化 OpenAI 客户端失败: {e}")
    sys.exit(1)

# 拿到项目根 = extractor 脚本的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"项目根目录: {PROJECT_ROOT}")

# public 目录的绝对路径
SCREENSHOT_DIR = PROJECT_ROOT / "public"
print(f"截图目录: {SCREENSHOT_DIR}")

def get_screenshot_paths():
    # 确保目录存在
    if not SCREENSHOT_DIR.is_dir():
        print(f"错误: 找不到截图目录：{SCREENSHOT_DIR}")
        return []

    screenshots = [
        str(SCREENSHOT_DIR / file)
        for file in SCREENSHOT_DIR.iterdir()
        if file.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    
    if not screenshots:
        print(f"警告: 在 {SCREENSHOT_DIR} 目录中没有找到任何图片文件")
    else:
        print(f"找到 {len(screenshots)} 个截图文件:")
        for path in screenshots:
            print(f"  - {path}")
    
    return screenshots


try:
    # Main execution flow
    print("开始执行主流程")
    screenshot_paths = get_screenshot_paths()
    if not screenshot_paths:
        print("没有找到任何截图，退出程序")
        sys.exit(1)
        
    item_info = []

    # Process each screenshot
    for path in screenshot_paths:
        print(f"处理截图: {path}")
        try:
            image = Image.open(path)
            print(f"成功打开图片，大小: {image.size}")
            text = pytesseract.image_to_string(image, lang='eng')
            print(f"OCR 文本提取完成，文本长度: {len(text)}")
            print(f"OCR 文本样例 (前100个字符): {text[:100]}")

            # Use OpenAI API to analyze extracted text
            print("调用 OpenAI API 分析文本...")
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
            print(f"API 返回内容样例 (前100个字符): {gpt_response[:100]}")

            # Clean response
            cleaned_response = re.sub(r'```json|```', '', gpt_response).strip()

            # Try to load JSON data
            try:
                extracted_info = json.loads(cleaned_response)
                print(f"成功解析 JSON 数据，类型: {type(extracted_info)}")
                if isinstance(extracted_info, list):
                    item_info.extend(extracted_info)
                    print(f"从该图片中提取了 {len(extracted_info)} 条项目信息")
                else:
                    print(f"警告: 预期得到列表，但得到了 {type(extracted_info)}")
            except json.JSONDecodeError as je:
                print(f"错误: 无法解码 JSON，原始响应: {cleaned_response}")
                print(f"JSON 解码错误: {je}")
                continue
                
        except Exception as e:
            print(f"处理截图 {path} 时出错: {e}")
            traceback.print_exc()
            continue

    # Save item_info to item_info.json
    if item_info:
        output_path = Path('item_info.json')
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(item_info, json_file, ensure_ascii=False, indent=4)
        print(f"所有截图已处理完毕，项目信息已保存到 {output_path.absolute()}")
    else:
        print("没有从截图中提取到有效项目信息")

except Exception as e:
    print(f"程序执行过程中出现错误: {e}")
    traceback.print_exc()
    sys.exit(1)