# server.py  ——  FastMCP server entry point
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
import sys
from extractor.pipeline import run_script, SCRIPT_DIR
from extractor.DrissionPage_Downloader import OUTPUT_DIR, CONFIG, generate_mhtml_filename
from DrissionPage import Chromium, ChromiumOptions  # type: ignore
from playwright.async_api import async_playwright, Browser, Page  # type: ignore
from dotenv import load_dotenv
import asyncio
import os
import config
from extractor import ocr  # import OCR module
from extractor.Tag_Locating import process_name_tag_location, process_price_tag_location  # import tag locating module
from extractor.Final_Summary import process_final_summary  # import the encapsulated process_final_summary function
from extractor.css_selector_generator import process_extraction_request, process_natural_language_request  # import CSS selector generator
from extractor.extraction_executor import execute_extraction  # import extraction executor
import json
import openai

def debug(msg: str):
    # 所有调试信息都打印到 stderr，避免干扰 stdio JSON-RPC 流
    print(msg, file=sys.stderr)

# 加载环境变量
load_dotenv()

debug("== InfoExtractor server starting ==")

# 全局浏览器变量
BROWSER: Chromium | None = None
PLAYWRIGHT_BROWSER: Browser | None = None
PLAYWRIGHT = None

def get_browser() -> Chromium:
    """
    初始化并返回全局 Chromium 实例。后续调用都会复用同一个浏览器。
    """
    global BROWSER
    if BROWSER is None:
        opts = ChromiumOptions()
        opts.incognito()
        opts.auto_port(True)
        BROWSER = Chromium(addr_or_opts=opts)
        debug(">> Chromium browser launched")
    return BROWSER

async def get_playwright_browser() -> Browser:
    """
    初始化并返回全局 Playwright Browser 实例。
    """
    global PLAYWRIGHT_BROWSER, PLAYWRIGHT
    if PLAYWRIGHT_BROWSER is None:
        PLAYWRIGHT = await async_playwright().start()
        PLAYWRIGHT_BROWSER = await PLAYWRIGHT.chromium.launch(
            headless=True,
        )
        debug(">> Playwright browser launched")
    return PLAYWRIGHT_BROWSER

def quit_browser():
    """
    关闭全局浏览器实例。
    """
    global BROWSER
    if BROWSER is not None:
        debug(">> Quitting Chromium browser")
        BROWSER.quit()
        BROWSER = None

async def quit_playwright_browser():
    """
    关闭全局 Playwright 浏览器实例。
    """
    global PLAYWRIGHT_BROWSER, PLAYWRIGHT
    if PLAYWRIGHT_BROWSER is not None:
        debug(">> Quitting Playwright browser")
        await PLAYWRIGHT_BROWSER.close()
        PLAYWRIGHT_BROWSER = None
    if PLAYWRIGHT is not None:
        await PLAYWRIGHT.stop()
        PLAYWRIGHT = None

@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[dict]:
    # 启动时初始化所有依赖环境
    debug(">> lifespan: initializing resources")
    # get_browser()
    await get_playwright_browser()
    yield {}
    debug(">> lifespan: cleaning up resources")
    quit_browser()
    await quit_playwright_browser()
    debug(">> All browsers quit")

mcp = FastMCP(
    "InfoExtractor",
    lifespan=lifespan,
    dependencies=[
        # List dependencies here, uv/pip will install automatically
        "drissionpage",
        "beautifulsoup4",
        "pytesseract",
        "python-dotenv",
        "openai",
        "playwright"
    ],
)

# -----------------------------------
# Tool 1: Download URL and save as MHTML
# -----------------------------------
@mcp.tool()
async def download_urls_tool(
    urls: list[str],
    *,
    ctx: Context
) -> dict:
    """
    This is the first step for extracting process.
    Next you need to call the Screenshot Tool.
    Download Tool: Download the given URLs and save them as .mhtml files in the output directory.
    """
    debug(f"--> download_urls_tool called with: {urls}")
    browser = get_browser()
    saved_paths: dict[str, str] = {}

    for url in urls:
        try:
            debug(f"----> Downloading {url}")
            tab = browser.new_tab()
            tab.get(url)
            tab.wait(CONFIG["waitTime"])

            filename = generate_mhtml_filename(url)
            tab.save(path=str(OUTPUT_DIR), name=filename[:-6], as_pdf=False)
            tab.close()

            full_path = str(OUTPUT_DIR / filename)
            saved_paths[url] = full_path
            debug(f"----> Saved to {full_path}")
        except Exception as e:
            debug(f"!!!! download failed for {url}: {e}")
            saved_paths[url] = f"ERROR: {e}"

    debug(f"<-- download_urls_tool result: {saved_paths}")
    return {"mhtml_files": saved_paths}

# -----------------------------------
# Tool 2: Screenshot
# -----------------------------------
@mcp.tool()
async def screenshot_tool(
    *,
    ctx: Context
) -> dict:
    """
    This is the second step for extracting process,
    product_name_processing_tool or product_price_processing_tool should be called next.
    Screenshot Tool: Take screenshots of all downloaded .mhtml files and save them in the public directory.
    """
    await ctx.info("📸 Running screenshot tool on all .mhtml files")

    # Ensure public directory exists
    public_dir = Path(__file__).parent / "public"
    public_dir.mkdir(exist_ok=True)

    results: dict[str, bool] = {}
    mhtml_files = list(OUTPUT_DIR.glob("*.mhtml"))
    if not mhtml_files:
        await ctx.info("⚠️ No .mhtml files found")
        return {"screenshots": results}

    browser = await get_playwright_browser()
    
    for path in mhtml_files:
        try:
            await ctx.info(f"📸 Processing {path.name}")
            
            # 创建新页面
            page = await browser.new_page()
            try:
                # 加载本地mhtml文件
                await page.goto(f"file://{path.resolve()}")
                
                # 等待页面加载完成
                await page.wait_for_load_state("networkidle")
                
                # 获取页面尺寸并设置视口
                viewport_size = await page.evaluate("""() => {
                    return {
                        width: Math.max(document.documentElement.clientWidth, window.innerWidth || 0),
                        height: Math.max(document.documentElement.clientHeight, window.innerHeight || 0)
                    }
                }""")
                
                await page.set_viewport_size(viewport_size)
                
                # 截图保存到public目录
                screenshot_path = public_dir / path.with_suffix(".png").name
                await page.screenshot(path=str(screenshot_path), full_page=True)
                
                await ctx.info(f"✅ Screenshot saved to {screenshot_path.absolute()}")
                success = True
                
            except Exception as e:
                await ctx.error(f"Screenshot failed for {path.name}: {str(e)}")
                success = False
            finally:
                await page.close()
                
        except Exception as e:
            await ctx.error(f"Unexpected error processing {path.name}: {str(e)}")
            success = False
            
        results[path.name] = success
        await ctx.info(f"{'✅' if success else '❌'} Finished {path.name}")

    await ctx.info("✅ All screenshots done")
    return {"screenshots": results}


# -----------------------------------
# Tool 3-4: Combined Tools: Product Name and Price Processing
# -----------------------------------
@mcp.tool()
async def product_name_processing_tool(
    *,
    ctx: Context
) -> dict:
    """
    This is the third step for extracting process.
    For information extraction, it is recommended to first extract product names, if product_name_processing_tool fails,
    you can use product_price_processing_tool to extract product prices.
    This tool will help to identify the target HTML blocks, and assist with extract_data_tool later.
    So if this tool succeeds, you can directly call extract_data_tool to get extraction schemas.
    Combined tool: Extract product name information from screenshots with OCR and locate name tags
    """
    debug("--> product_name_processing_tool called")
    await ctx.info("🔢 Running OCR and tag locating for product names")
    
    # Clear item_info.json file if exists
    item_info_path = Path("item_info.json")
    if item_info_path.exists():
        item_info_path.unlink()
    # Clear BeautifulSoup_Content.json file if exists
    beautifulsoup_content_path = Path("BeautifulSoup_Content.json")
    if beautifulsoup_content_path.exists():
        beautifulsoup_content_path.unlink()
        
    # Step 1: Execute OCR for product names
    await ctx.info("📸 Step 1: Extracting product names from screenshots using OCR...")
    ocr_success = await ocr.process_ocr_name(ctx)
    if not ocr_success:
        await ctx.error("❌ OCR for product names failed")
        return {"success": False, "step_completed": "ocr"}
    await ctx.info("✅ OCR for product names completed successfully")
    
    # Step 2: Locate product name tags
    await ctx.info("🏷️ Step 2: Locating product name tags in HTML...")
    # Clear BeautifulSoup_Content.json file if exists
    beautifulsoup_content_path = Path("BeautifulSoup_Content.json")
    if beautifulsoup_content_path.exists():
        beautifulsoup_content_path.unlink()
    tag_success = await process_name_tag_location(ctx)
    if not tag_success:
        await ctx.error("❌ Product name tag locating failed")
        return {"success": False, "step_completed": "ocr_only"}
    await ctx.info("✅ Product name tag locating completed successfully")
    
    return {"success": True, "step_completed": "both"}

@mcp.tool()
async def product_price_processing_tool(
    *,
    ctx: Context
) -> dict:
    """
    This is the alternative third step for extracting process. If product_name_processing_tool failed, 
    you can use this tool to extract product price information.
    Next you can call extract_data_tool to get extraction schemas.
    Combined tool: Extract price information from screenshots with OCR and locate price tags
    """
    debug("--> product_price_processing_tool called")
    await ctx.info("💲 Running OCR and tag locating for prices")

    # Clear item_info.json file if exists
    item_info_path = Path("item_info.json")
    if item_info_path.exists():
        item_info_path.unlink()
    # Clear BeautifulSoup_Content.json file if exists
    beautifulsoup_content_path = Path("BeautifulSoup_Content.json")
    if beautifulsoup_content_path.exists():
        beautifulsoup_content_path.unlink()
    
    # Step 1: Execute OCR for prices
    await ctx.info("📸 Step 1: Extracting price information from screenshots using OCR...")
    ocr_success = await ocr.process_ocr_price(ctx)
    if not ocr_success:
        await ctx.error("❌ OCR for prices failed")
        return {"success": False, "step_completed": "none"}
    await ctx.info("✅ OCR for prices completed successfully")
    
    # Step 2: Locate price tags
    await ctx.info("💲 Step 2: Locating price tags in HTML...")
    tag_success = await process_price_tag_location(ctx)
    if not tag_success:
        await ctx.error("❌ Price tag locating failed")
        return {"success": False, "step_completed": "ocr_only"}
    await ctx.info("✅ Price tag locating completed successfully")
    
    return {"success": True, "step_completed": "both"}

# -----------------------------------
# Tool 5: Intelligent Data Extraction Configuration Tool
# -----------------------------------
@mcp.tool()
async def extract_data_tool(
    extraction_request: str,
    *,
    ctx: Context
) -> dict:
    """
    This is the fourth step for extracting process.
    Next you can call execute_extraction_tool to perform data extraction.
    Intelligent Data Extraction Configuration Tool: Automatically generate extraction configuration based on natural language description
    Receives a natural language extraction request and generates a CSS selector configuration for data extraction.
    
    Args:
        extraction_request: Extraction requirement in natural language, e.g. "I want to extract all product names and prices"
    
    Returns:
        Dictionary containing CSS selector configuration
    """
    debug(f"--> extract_data_tool called with: {extraction_request}")
    await ctx.info("🧠 Processing extraction request...")
    try:
        # Call the natural language processing function to generate extraction configuration
        config_result = await process_natural_language_request(extraction_request)
        if "error" in config_result:
            await ctx.error(f"Extraction configuration generation failed: {config_result['error']}")
            return {"success": False, "error": config_result["error"]}
        # Get extraction configuration and save path
        selectors_config = config_result.get("selectors_config", {})
        schema_path = config_result.get("schema_path", "")
        # Output result info
        await ctx.info(f"✅ Extraction configuration generated")
        await ctx.info(f"📋 Website type: {selectors_config.get('website_type', 'Not specified')}")
        await ctx.info(f"📝 Description: {selectors_config.get('description', 'Not provided')}")
        # Output extraction field info
        fields = selectors_config.get("expected_fields", [])
        if fields:
            field_names = [field.get("name", "") for field in fields]
            await ctx.info(f"🔍 Extraction fields: {', '.join(field_names)}")
        # Show container selector info
        container_selector = selectors_config.get("container_selector", "")
        if container_selector:
            await ctx.info(f"🧩 Container selector: {container_selector}")
        await ctx.info(f"💾 Extraction configuration saved to: {schema_path}")
        # Return only the configuration result
        return {
            "success": True,
            "selectors_config": selectors_config,
            "schema_path": str(schema_path),
        }
    except Exception as e:
        debug(f"Extract data tool error: {str(e)}")
        await ctx.error(f"Extraction configuration generation error: {str(e)}")
        return {"success": False, "error": str(e)}

# -----------------------------------
# Tool 6: Execute Data Extraction Tool
# -----------------------------------
@mcp.tool()
async def execute_extraction_tool(
    selectors_config_path: str = "",
    *,
    ctx: Context
) -> dict:
    """
    This is the final step for extracting process.
    Execute Data Extraction Tool: Use the generated selector configuration to extract data from mhtml files
    
    Args:
        selectors_config_path: Selector configuration file path, if empty, use the latest configuration file
    
    Returns:
        Dictionary containing extraction results
    """
    debug(f"--> execute_extraction_tool called with config path: {selectors_config_path}")
    await ctx.info("⚙️ Starting data extraction...")
    try:
        # If no config path is provided, find the latest config file
        if not selectors_config_path:
            schemas_dir = Path("extraction_schemas")
            if schemas_dir.exists() and schemas_dir.is_dir():
                config_files = list(schemas_dir.glob("selector_schema_*.json"))
                if config_files:
                    latest_config = max(config_files, key=lambda p: p.stat().st_mtime)
                    selectors_config_path = str(latest_config)
                    await ctx.info(f"📄 Using latest config file: {latest_config.name}")
                else:
                    await ctx.error("❌ No config files found")
                    return {"success": False, "error": "No config files found"}
            else:
                await ctx.error("❌ Config directory does not exist")
                return {"success": False, "error": "Config directory does not exist"}
        # Get browser instance
        browser = await get_playwright_browser()
        # Call extraction executor to perform extraction task
        # Pass info_callback and error_callback so the executor can send messages to the user
        result = await execute_extraction(
            browser=browser,
            selectors_config_path=selectors_config_path,
            info_callback=ctx.info,
            error_callback=ctx.error
        )
        return result
    except Exception as e:
        debug(f"Execute extraction tool error: {str(e)}")
        await ctx.error(f"Data extraction error: {str(e)}")
        return {"success": False, "error": str(e)}



if __name__ == "__main__":
    debug("== entering mcp.run() ==")
    mcp.run()
    debug("== mcp.run() has exited ==")
