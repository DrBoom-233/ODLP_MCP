"""
pipeline.py
-----------
Tool-1  download_urls(urls):   仅下载 URL → 保存到 mhtml_output/
Tool-2  extract_mhtml():       遍历 mhtml_output/*.mhtml → OCR/LLM → 输出价格 JSON
"""

from __future__ import annotations
import sys
import asyncio
import contextlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ────────────────────────────────────────────────
# 路径常量
# ────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent        # mcp-project/
SCRIPT_DIR = Path(__file__).resolve().parent               # mcp-project/extractor/

MHTML_DIR  = ROOT_DIR / "mhtml_output"
PRICE_DIR  = ROOT_DIR / "price_info_output"
JSON_FILES = [
    SCRIPT_DIR / "item_info.json",
    SCRIPT_DIR / "BeautifulSoup_Content.json",
    SCRIPT_DIR / "price_info.json",
]

# 初始化目录与空 JSON
for d in (MHTML_DIR, PRICE_DIR):
    d.mkdir(parents=True, exist_ok=True)

for jf in JSON_FILES:
    if not jf.exists():
        jf.write_text("{}", encoding="utf-8")

# ────────────────────────────────────────────────
# 日志函数：即使 ctx=None 也会打印到 stderr
# ────────────────────────────────────────────────

def _log(msg: str, ctx: "mcp.server.fastmcp.Context | None" = None) -> None:
    if ctx:
        ctx.info(msg)
    else:
        print(msg, file=sys.stderr)

# ────────────────────────────────────────────────
# 通用工具
# ────────────────────────────────────────────────
def clear_json_files() -> None:
    for p in JSON_FILES:
        p.write_text("{}", encoding="utf-8")


async def run_sub(cmd: List[str]) -> Tuple[str, str]:
    """async-subprocess 封装，返回 (stdout, stderr)"""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = await proc.communicate()
    return out.decode(errors="replace"), err.decode(errors="replace")


async def run_script(
    script: Tuple[str, str, bool],
    mhtml: str | None = None,
    *,
    ctx: "mcp.server.fastmcp.Context | None" = None,
) -> bool:
    exe, rel_path, need_file = script
    cmd = [exe, str(SCRIPT_DIR / rel_path)]

    # 如果需要文件参数，就拼出 mhtml 的全路径
    if need_file and mhtml:
        file_path = Path(mhtml)
        if not file_path.is_absolute():
            file_path = MHTML_DIR / file_path
        file_path = file_path.resolve()
        if ctx:
            ctx.info(f"▶ Using MHTML file: {file_path}")
        cmd.append(str(file_path))

    if ctx:
        ctx.info(f"▶ Running command: {' '.join(cmd)}")
    out, err = await run_sub(cmd)

    if out and ctx:
        ctx.info(out[:300])
    if err:
        raise RuntimeError(err[:500])
    return True

# ────────────────────────────────────────────────
# 脚本清单
# ────────────────────────────────────────────────
MAIN_SCRIPTS = [
    ("python", "screenshot.py",        True),
    ("python", "image_transform_2.py", False),
    ("python", "Tag_Locating_2.py",    True),
    ("python", "Final_Summary.py",     False),
]
FALLBACK_SCRIPTS = [
    ("python", "image_transform.py", False),
    ("python", "Tag_Locating.py",    True),
    ("python", "Final_Summary.py",   False),
]

# ────────────────────────────────────────────────
# 核心处理：单个 MHTML
# ────────────────────────────────────────────────
async def process_one(
    mhtml_path: Path, *, ctx: "mcp.server.fastmcp.Context | None" = None
) -> Dict[str, Any]:
    if ctx:
        ctx.info(f"📄 Processing file: {mhtml_path}")
    clear_json_files()

    for s in MAIN_SCRIPTS:
        await run_script(s, str(mhtml_path), ctx=ctx)

    price_path = SCRIPT_DIR / "price_info.json"
    data = json.loads(price_path.read_text(encoding="utf-8") or "{}")
    if data:
        return data

    for s in FALLBACK_SCRIPTS:
        await run_script(s, str(mhtml_path), ctx=ctx)

    return json.loads(price_path.read_text(encoding="utf-8") or "{}")

# ────────────────────────────────────────────────
# 下载器
# ────────────────────────────────────────────────
async def download_mhtml(urls: List[str], *, ctx=None) -> None:
    if not urls:
        return
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "DrissionPage_Downloader.py"),
        "--urls",
        *urls,
    ]
    if ctx:
        ctx.info("⬇️  " + " ".join(cmd))
    out, err = await run_sub(cmd)
    if err:
        raise RuntimeError("Downloader error: " + err[:300])
    if ctx:
        ctx.info(out[:300])

# ────────────────────────────────────────────────
# 心跳协程
# ────────────────────────────────────────────────
async def _heartbeat(ctx, interval: int = 10):
    tick = 0
    while True:
        await asyncio.sleep(interval)
        tick += 1
        if ctx:
            ctx.info(f"⏳ still running… {tick * interval}s elapsed")

# ────────────────────────────────────────────────
# Tool-1: 仅下载 URL
# ────────────────────────────────────────────────
async def download_urls(
    urls: List[str],
    *,
    ctx: "mcp.server.fastmcp.Context | None" = None,
) -> Dict[str, Any]:
    # 第一行：打印收到的 urls，帮助调试
    if ctx:
        ctx.info(f"🛠️  download_urls called with urls = {urls}")

    # 然后启动心跳防超时
    hb = asyncio.create_task(_heartbeat(ctx)) if ctx else None
    try:
        if not urls:
            raise ValueError("urls list is empty")
        await download_mhtml(urls, ctx=ctx)

        saved = sorted(MHTML_DIR.glob("*.mhtml"), key=lambda p: p.stat().st_mtime, reverse=True)[: len(urls)]
        files = [p.name for p in saved]
        if ctx:
            ctx.info(f"✅ Downloaded {len(files)} files")
        return {"files": files}
    finally:
        if hb:
            hb.cancel()

# ────────────────────────────────────────────────
# Tool-2: 处理现有 MHTML
# ────────────────────────────────────────────────
async def extract_mhtml(
    *,
    ctx: "mcp.server.fastmcp.Context | None" = None,
) -> Dict[str, Any]:
    if ctx:
        await ctx.info(f"🔍 pipeline.extract_mhtml start — MHTML_DIR={MHTML_DIR}")
    hb = asyncio.create_task(_heartbeat(ctx)) if ctx else None
    try:
        mhtml_files = sorted(
            MHTML_DIR.glob("*.mhtml"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if ctx:
            ctx.info(f"🔎 Looking in: {MHTML_DIR}")
            ctx.info(f"🔎 Found {len(mhtml_files)} .mhtml: {[p.name for p in mhtml_files]}")
        if not mhtml_files:
            raise FileNotFoundError("No MHTML to process")

        results: Dict[str, Any] = {}
        total = len(mhtml_files)
        for idx, path in enumerate(mhtml_files, 1):
            if ctx:
                ctx.report_progress(idx, total)
                ctx.info(f"🔍 [{idx}/{total}] {path.name}")
            res = await process_one(path, ctx=ctx)
            results[path.name] = res
            # 正确写法：用 f-string 生成文件名
            output_file = PRICE_DIR / f"{path.stem}.json"
            output_file.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")

        if ctx:
            ctx.info("✅ Extraction finished")
        return results
    finally:
        if hb:
            hb.cancel()
