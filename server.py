# server.py  ——  FastMCP server 入口
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
from extractor import ocr  # 导入OCR模块
from extractor.Tag_Locating import process_name_tag_location, process_price_tag_location  # 导入标签定位模块
from extractor.Final_Summary import process_final_summary  # 导入封装好的process_final_summary函数

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
    get_browser()
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
        # 在这里列依赖，uv/pip 会自动安装
        "drissionpage",
        "beautifulsoup4",
        "pytesseract",
        "python-dotenv",
        "openai",
        "playwright"
    ],
)

# -----------------------------------
# Tool 1: 下载 URL 并保存为 MHTML
# -----------------------------------
@mcp.tool()
async def download_urls_tool(
    urls: list[str],
    *,
    ctx: Context
) -> dict:
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
# Tool 2: 截图
# -----------------------------------
@mcp.tool()
async def screenshot_tool(
    *,
    ctx: Context
) -> dict:
    await ctx.info("📸 Running screenshot tool on all .mhtml files")

    # 确保public目录存在
    public_dir = Path("public")
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
                
                await ctx.info(f"✅ Screenshot saved to {screenshot_path}")
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
# Tool 3: OCR 转换 (image_transform)
# -----------------------------------
@mcp.tool()
async def ocr_name_tool(
        *,
        ctx: Context
) -> dict:
    """
    OCR工具1：从截图中提取商品名称信息
    """
    debug("--> ocr_name_tool called")
    await ctx.info("🔢 Running OCR for Item Names")
    
    success = await ocr.process_ocr_name(ctx)
    
    await ctx.info(f"商品名称OCR工具执行结果: {'成功' if success else '失败'}")
    return {"success": success}

@mcp.tool()
async def ocr_price_tool(
        *,
        ctx: Context
) -> dict:
    """
    OCR工具2：从截图中提取价格信息
    """
    debug("--> ocr_price_tool called")
    await ctx.info("💲 Running OCR for Item Prices")
    
    success = await ocr.process_ocr_price(ctx)
    
    await ctx.info(f"商品价格OCR工具执行结果: {'成功' if success else '失败'}")
    return {"success": success}

# -----------------------------------
# Tool 4: 标签定位 (Tag Locating)
# -----------------------------------
@mcp.tool()
async def name_tag_locating_tool(
    *,
    ctx: Context
) -> dict:
    """
    标签定位工具1：定位商品名称标签
    """
    debug("--> name_tag_locating_tool called")
    await ctx.info("🏷️ 开始定位商品名称标签")
    
    success = await process_name_tag_location(ctx)
    
    await ctx.info(f"商品名称标签定位结果: {'成功' if success else '失败'}")
    return {"success": success}

@mcp.tool()
async def price_tag_locating_tool(
    *,
    ctx: Context
) -> dict:
    """
    标签定位工具2：定位商品价格标签
    """
    debug("--> price_tag_locating_tool called")
    await ctx.info("💲 开始定位商品价格标签")
    
    success = await process_price_tag_location(ctx)
    
    await ctx.info(f"商品价格标签定位结果: {'成功' if success else '失败'}")
    return {"success": success}

# -----------------------------------
# Tool 5: 最终摘要 (Final Summary)
# -----------------------------------
@mcp.tool()
async def final_summary_tool(
    *,
    ctx: Context
) -> dict:
    debug("--> final_summary_tool called")
    await ctx.info("📝 Running final summary tool")
    
    # 直接调用封装好的函数，而不是运行Python脚本
    try:
        success = process_final_summary()
        if success:
            await ctx.info("✅ 最终摘要已完成，并已生成price_info.json文件")
        else:
            await ctx.info("❌ 最终摘要处理失败")
    except Exception as e:
        debug(f"Final summary processing error: {str(e)}")
        await ctx.error(f"最终摘要处理出错: {str(e)}")
        success = False
    
    return {"summary_ok": success}

if __name__ == "__main__":
    debug("== entering mcp.run() ==")
    mcp.run()
    debug("== mcp.run() has exited ==")
