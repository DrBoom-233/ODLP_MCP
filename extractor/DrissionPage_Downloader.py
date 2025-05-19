"""
DrissionPage_Downloader.py
-------------------------
• 输出目录统一到 *项目根/mhtml_output/*，与 pipeline.py 保持一致。
• 支持两种调用方式：
  1) CLI 传 `--urls ...`（与之前兼容）
  2) 若 CLI 不传任何 URL，则使用脚本顶部 `DEFAULT_URLS` 进行快速测试；要停用只需把列表留空或注释掉。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from DrissionPage import Chromium, ChromiumOptions  # type: ignore

# --------------------------------------------------------
# **硬编码测试列表**：开发自测时使用。正式运行可留空或注释
# --------------------------------------------------------
DEFAULT_URLS: list[str] = [
     # "https://www.foodbasics.ca/aisles/fruits-vegetables/fruits",
]

# --------------------------------------------------------
# 路径常量：输出目录固定在项目根的 mhtml_output/
# --------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent     # mcp-project/
OUTPUT_DIR = ROOT_DIR / "mhtml_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "waitTime": 3,   # 页面加载等待秒数
}

# --------------------------------------------------------
# 工具函数
# --------------------------------------------------------

def generate_mhtml_filename(url: str) -> str:
    """根据 URL 生成文件名：host_最后路径片段_日期.mhtml"""
    parsed = urlparse(url)
    host = parsed.hostname.replace(".", "_") if parsed.hostname else "unknown"
    last_seg = Path(parsed.path).name or "no_category"
    date_str = time.strftime("%Y%m%d")
    return f"{host}_{last_seg}_{date_str}.mhtml"


def download_single(url: str) -> None:
    print(f"\n=== Download: {url} ===", file=sys.stderr)
    co = ChromiumOptions()
    co.incognito()
    browser = Chromium(addr_or_opts=co)
    tab = browser.latest_tab
    tab.get(url)
    tab.wait(CONFIG["waitTime"])

    filename = generate_mhtml_filename(url)
    tab.save(path=str(OUTPUT_DIR), name=filename[:-6], as_pdf=False)
    browser.quit()
    print(f"✅ Saved → {OUTPUT_DIR / filename}", file=sys.stderr)


# --------------------------------------------------------
# CLI 入口
# --------------------------------------------------------

def parse_cli() -> list[str]:
    parser = argparse.ArgumentParser(
        description="Save given URLs as .mhtml via DrissionPage")
    parser.add_argument("--urls", nargs="*", help="URLs to download")
    args = parser.parse_args()
    return args.urls or []


def main() -> None:
    urls = parse_cli()

    if not urls:
        if DEFAULT_URLS:
            print("[Downloader] No --urls provided, fallback to DEFAULT_URLS", file=sys.stderr)
            urls = DEFAULT_URLS
        else:
            print("[Downloader] No URL supplied. Exit.", file=sys.stderr)
            sys.exit(0)

    for u in urls:
        try:
            download_single(u)
        except Exception as e:
            print(f"❌ Failed {u}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
