"""
DrissionPage_Downloader.py
-------------------------
• Output directory unified to *project_root/mhtml_output/*, consistent with pipeline.py.
• Supports two invocation methods:
  1) CLI passes `--urls ...` (compatible with previous versions)
  2) If no URL is passed via CLI, the `DEFAULT_URLS` at the top of the script is used for quick testing; to disable, leave the list empty or comment it out.
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
# **Hardcoded test list**: Used for development self-testing. Leave empty or comment out for official runs.
# --------------------------------------------------------
DEFAULT_URLS: list[str] = [
     # "https://www.foodbasics.ca/aisles/fruits-vegetables/fruits",
]

# --------------------------------------------------------
# Path constants: Output directory fixed at project root's mhtml_output/
# --------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parent.parent     # mcp-project/
OUTPUT_DIR = ROOT_DIR / "mhtml_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "waitTime": 3,   # Page load wait time in seconds
}

# --------------------------------------------------------
# Utility functions
# --------------------------------------------------------

def generate_mhtml_filename(url: str) -> str:
    """Generate a filename based on the URL: host_last_path_segment_date.mhtml"""
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
# CLI entry point
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
