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
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

"""
This file is not longer used in the main workflow. Please ignore it.
"""


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
    ctx: "mcp.server.fastmcp.Context | None" = None,) -> bool:
    exe, rel_path, need_file = script
    cmd = [exe, str(SCRIPT_DIR / rel_path)]

    script_full_path = SCRIPT_DIR / rel_path
    if not script_full_path.exists():
        error_msg = f"脚本文件不存在: {script_full_path}"
        if ctx:
            ctx.error(error_msg)
        else:
            print(error_msg, file=sys.stderr)
        return False

    # 打印当前工作目录和脚本所在目录
    cwd_msg = f"当前工作目录: {Path().absolute()}"
    script_dir_msg = f"脚本所在目录: {SCRIPT_DIR.absolute()}"
    if ctx:
        ctx.info(cwd_msg)
        ctx.info(script_dir_msg)
    else:
        print(cwd_msg, file=sys.stderr)
        print(script_dir_msg, file=sys.stderr)

    # 如果需要文件参数，就拼出 mhtml 的全路径
    if need_file and mhtml:
        file_path = Path(mhtml)
        if not file_path.is_absolute():
            file_path = MHTML_DIR / file_path
        file_path = file_path.resolve()
        if not file_path.exists():
            error_msg = f"目标文件不存在: {file_path}"
            if ctx:
                ctx.error(error_msg)
            else:
                print(error_msg, file=sys.stderr)
            return False
            
        if ctx:
            ctx.info(f"▶ Using MHTML file: {file_path}")
        cmd.append(str(file_path))

    # 添加环境变量
    env = os.environ.copy()
    
    # 确保 PYTHONPATH 包含当前目录和项目根目录
    python_path = env.get('PYTHONPATH', '')
    if python_path:
        env['PYTHONPATH'] = f"{ROOT_DIR}{os.pathsep}{python_path}"
    else:
        env['PYTHONPATH'] = str(ROOT_DIR)
    
    # 打印命令和环境变量
    if ctx:
        ctx.info(f"▶ Running command: {' '.join(cmd)}")
        ctx.info(f"▶ PYTHONPATH: {env['PYTHONPATH']}")
    else:
        print(f"▶ Running command: {' '.join(cmd)}", file=sys.stderr)
        print(f"▶ PYTHONPATH: {env['PYTHONPATH']}", file=sys.stderr)
    
    try:
        # 使用带环境变量的异步子进程
        proc = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(ROOT_DIR)  # 显式设置工作目录为项目根目录
        )
        
        # 添加超时
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            out_str = out.decode(errors="replace")
            err_str = err.decode(errors="replace")
            
            # 打印输出
            if out_str:
                out_msg = f"脚本输出: {out_str[:500]}"
                if ctx:
                    ctx.info(out_msg)
                else:
                    print(out_msg, file=sys.stderr)
                    
            # 检查错误
            if err_str:
                err_msg = f"脚本错误: {err_str[:500]}"
                if ctx:
                    ctx.error(err_msg)
                else:
                    print(err_msg, file=sys.stderr)
                if proc.returncode != 0:
                    return False
                    
            return proc.returncode == 0
            
        except asyncio.TimeoutError:
            error_msg = f"脚本执行超时: {rel_path}"
            if ctx:
                ctx.error(error_msg)
            else:
                print(error_msg, file=sys.stderr)
                
            # 尝试终止进程
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except:
                proc.kill()
            return False
            
    except Exception as e:
        error_msg = f"执行脚本时发生异常: {e}"
        if ctx:
            ctx.error(error_msg)
        else:
            print(error_msg, file=sys.stderr)
        import traceback
        trace_msg = traceback.format_exc()
        if ctx:
            ctx.error(trace_msg)
        else:
            print(trace_msg, file=sys.stderr)
        return False

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
