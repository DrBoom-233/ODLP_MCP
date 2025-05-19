# server.py  ——  FastMCP server 入口
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP, Context
import sys
from extractor.DrissionPage_Downloader import OUTPUT_DIR, CONFIG, generate_mhtml_filename
from DrissionPage import Chromium, ChromiumOptions  # type: ignore
from dotenv import load_dotenv
import pytesseract
import subprocess


def debug(msg: str):
    # 所有调试信息都打印到 stderr，避免干扰 stdio JSON-RPC 流
    print(msg, file=sys.stderr)

debug("== InfoExtractor server starting ==")

# 全局浏览器变量
BROWSER: Chromium | None = None

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

def quit_browser():
    """
    关闭全局浏览器实例。
    """
    global BROWSER
    if BROWSER is not None:
        debug(">> Quitting Chromium browser")
        BROWSER.quit()
        BROWSER = None

@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[dict]:
    # 启动时初始化所有依赖环境
    debug(">> lifespan: initializing resources")
    # 如果你希望一启动就初始化浏览器，可以在这里调用 get_browser()
    get_browser()
    yield {}
    debug(">> lifespan: cleaning up resources")
    quit_browser()
    debug(">> Chromium browser quit")

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
    ],
)

@mcp.tool()
async def download_urls_tool(
    urls: list[str],
    *,
    ctx: Context | None = None,
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

@mcp.tool()
async def extract_mhtml_tool(*, ctx: Context | None = None) -> dict:
    debug("--> extract_mhtml_tool called")
    from extractor.pipeline import extract_mhtml

    result = await extract_mhtml(ctx=ctx)

    if ctx:
        await ctx.info(f"✅ extract_mhtml_tool exit, got: {result}")

    # await the info coroutine
    await ctx.info("🚀 extract_mhtml_tool entry")
    debug(f"<-- extract_mhtml_tool result: {result}")
    return result

@mcp.tool()
async def test(ctx: Context) -> dict:
    await ctx.info("ping")
    return {"ok": True}


if __name__ == "__main__":
    debug("== entering mcp.run() ==")
    mcp.run()
    debug("== mcp.run() has exited ==")
