# server.py  ——  FastMCP server 入口
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP, Context
import sys
from extractor.pipeline import run_script, SCRIPT_DIR
from extractor.DrissionPage_Downloader import OUTPUT_DIR, CONFIG, generate_mhtml_filename
from DrissionPage import Chromium, ChromiumOptions  # type: ignore
from dotenv import load_dotenv
import asyncio

def debug(msg: str):
    # 所有调试信息都打印到 stderr，避免干扰 stdio JSON-RPC 流
    print(msg, file=sys.stderr)

# 加载环境变量
load_dotenv()

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
    # get_browser()
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

    results: dict[str, bool] = {}
    mhtml_files = list(OUTPUT_DIR.glob("*.mhtml"))
    if not mhtml_files:
        await ctx.info("⚠️ No .mhtml files found")
        return {"screenshots": results}

    for path in mhtml_files:
        await ctx.info(f"📸 Processing {path.name}")
        script = SCRIPT_DIR / "screenshot.py"
        cmd = [
            sys.executable,
            str(script),
            str(path.resolve())
        ]
        await ctx.info(f"▶ Running: {' '.join(cmd)}")

        # 异步启动子进程
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_bytes, err_bytes = await proc.communicate()
        out, err = out_bytes.decode(), err_bytes.decode()

        if out:
            # 只截前 500 字，避免太长
            await ctx.info(out[:500].strip())
        if err:
            await ctx.error(err[:500].strip())

        success = (proc.returncode == 0)
        results[path.name] = success

    await ctx.call_tool("ping_tool", {})
    await ctx.info("✅ All screenshots done")
    return {"screenshots": results}

# -----------------------------------
# Tool 3: OCR 转换 (image_transform)
# -----------------------------------
@mcp.tool()
async def ocr_tool(
    *,
    ctx: Context
) -> dict:
    debug("--> ocr_tool called")
    await ctx.info("🔢 Running OCR tool")
    # 先尝试新版脚本
    try:
        await ctx.info("Running image_transform_2.py")
        ok = await run_script(("python", "image_transform_2.py", False), None, ctx=ctx)
    except Exception:
        # 回退旧版脚本
        ok = await run_script(("python", "image_transform.py", False), None, ctx=ctx)
    return {"ocr_ok": ok}

# -----------------------------------
# Tool 4: 标签定位 (Tag Locating)
# -----------------------------------
@mcp.tool()
async def tag_locating_tool(
    mhtml_path: str,
    *,
    ctx: Context
) -> dict:
    debug(f"--> tag_locating_tool called on: {mhtml_path}")
    await ctx.info("🏷️ Running tag locating tool")
    try:
        ok = await run_script(("python", "Tag_Locating_2.py", True), mhtml_path, ctx=ctx)
    except Exception:
        ok = await run_script(("python", "Tag_Locating.py", True), mhtml_path, ctx=ctx)
    return {"tag_locating_ok": ok}

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
    success = await run_script(("python", "Final_Summary.py", False), None, ctx=ctx)
    return {"summary_ok": success}

if __name__ == "__main__":
    debug("== entering mcp.run() ==")
    mcp.run()
    debug("== mcp.run() has exited ==")
