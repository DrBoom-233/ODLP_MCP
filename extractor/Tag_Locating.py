"""
标签定位模块 - 提供从MHTML文件中定位商品名称和价格标签的功能
"""
import json
import os
import time
import random
import argparse
import sys
import io
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio

# 设置stdout和stderr的编码为utf-8，解决Windows下中文编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

# ────────────────────────────────────────────────────────────────────────────────
# 目录常量：基于脚本位置定位项目根和 mhtml 输出目录
# ────────────────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent         # extractor/
PROJECT_ROOT = THIS_DIR.parent                     # mcp-project 根目录
MHTML_DIR = PROJECT_ROOT / "mhtml_output"          # mhtml_output 与 extractor 同级

# ────────────────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────────────────
def get_item_paths(soup: BeautifulSoup, product_names: list[str]) -> dict[str, list]:
    """
    根据 product_names 查找在 DOM 中的路径并返回。
    """
    paths: dict[str, list] = defaultdict(list)
    for product in product_names:
        # 精确或部分匹配标签文本
        tag = soup.find(lambda tag: tag.string and product in tag.string)
        if tag:
            path = get_dom_path(tag)
            paths[path].append(tag)
    return paths

def get_dom_path(tag) -> str:
    """
    获取从当前标签到根节点的 DOM 路径。
    """
    segments = []
    while tag is not None:
        segments.append(tag.name)
        tag = tag.parent
    return " > ".join(reversed(segments))

def filter_paths(paths: dict[str, list]) -> list:
    """
    筛选出现次数最多的路径对应的标签列表，随机返回至多两个。
    """
    if not paths:
        return []
    # 计算每条路径对应标签列表的长度，取最大值
    max_occurrence = max(len(tags) for tags in paths.values())
    # 只保留出现次数等于最大值的路径
    filtered = {p: tags for p, tags in paths.items() if len(tags) == max_occurrence}
    candidate_tags = next(iter(filtered.values()), [])
    # 不足两个时直接返回
    if len(candidate_tags) <= 2:
        return candidate_tags
    return random.sample(candidate_tags, 2)

def find_parent_with_multiple_descriptions(tags: list) -> any:
    """
    在候选标签中找到最低公共父元素，要求它的子节点中包含所有标签文本。
    """
    if not tags:
        return None
    parents = [tag.parent for tag in tags]
    while True:
        # 如果所有父节点相同，且都包含所有描述，就返回它
        if all(parents[0] is p for p in parents):
            parent = parents[0]
            texts = [t.get_text() for t in tags]
            # 检查 parent 的所有后代是否包含这些文本
            if all(any(text in desc.get_text() for desc in parent.find_all()) for text in texts):
                return parent
        # 否则往上再找一层
        parents = [p.parent or p for p in parents]
        # 到了 html 根还没找到，就放弃
        if all(p.name == "html" for p in parents):
            return None

def get_mhtml_file(file_path: str | None = None) -> Path:
    """
    获取要处理的MHTML文件
    """
    if file_path:
        fp = Path(file_path)
    else:
        # 找项目根的 mhtml_output 下最新的 .mhtml
        fp = next(MHTML_DIR.glob("*.mhtml"), None)
    
    if not fp or not fp.exists():
        raise FileNotFoundError(f"找不到要处理的 MHTML 文件：{fp}")
    
    return fp

async def get_html_content(file_path: Path) -> str:
    """
    使用Playwright异步API加载MHTML文件并获取HTML内容
    """
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()

    # 构造 file:// URL
    file_url = file_path.as_uri()
    await page.goto(file_url)

    # 等待页面加载
    await asyncio.sleep(5)
    html_content = await page.content()
    await browser.close()
    await playwright.stop()
    
    return html_content

def save_beautiful_soup_content(beautiful_soup: List[Dict]) -> bool:
    """
    保存提取的内容到JSON文件
    """
    if beautiful_soup:
        out_path = THIS_DIR / "BeautifulSoup_Content.json"
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(beautiful_soup, jf, ensure_ascii=False, indent=4)
        print(f"已写入 JSON：{out_path}")
        return True
    else:
        print("公共父节点下没有有效子节点内容。")
        return False

async def load_item_info(ctx, key: str = 'item') -> List[str]:
    """
    从item_info.json加载信息
    key可以是'item'(商品名称)或'price'(价格)
    """
    product_names = []
    item_info_path = PROJECT_ROOT / 'item_info.json'
    
    if not item_info_path.exists():
        item_info_path = THIS_DIR / 'item_info.json'
    
    if not item_info_path.exists():
        await ctx.error(f"找不到item_info.json文件")
        return []
    
    try:
        with open(item_info_path, 'r', encoding='utf-8') as f:
            try:
                item_data = json.load(f)
                product_names = [str(item.get(key, '')) for item in item_data if key in item]
                await ctx.info(f"找到{len(product_names)}个{key}信息")
                return product_names
            except json.JSONDecodeError as e:
                await ctx.error(f"解析{item_info_path}失败: {str(e)}")
                return []
    except Exception as e:
        await ctx.error(f"读取{item_info_path}失败: {str(e)}")
        return []

# ────────────────────────────────────────────────────────────────────────────────
# 主流程函数
# ────────────────────────────────────────────────────────────────────────────────
async def process_tag_location(ctx, product_names: List[str], file_path: str | None = None) -> bool:
    """
    通用的标签定位处理流程(异步版本)
    """
    try:
        # 1. 获取MHTML文件
        fp = get_mhtml_file(file_path)
        await ctx.info(f"处理MHTML文件: {fp}")
        
        # 2. 用Playwright加载页面获取HTML内容
        html_content = await get_html_content(fp)
        
        # 3. 解析DOM，定位产品名称对应的标签路径
        soup = BeautifulSoup(html_content, "html.parser")
        paths = get_item_paths(soup, product_names)
        await ctx.info(f"找到的路径数量: {len(paths)}")
        
        # 4. 筛选出现次数最多的标签
        majority_tags = filter_paths(paths)
        if not majority_tags:
            await ctx.warning("没有匹配到有效的标签，跳过后续处理。")
            return False
        await ctx.info(f"选取了{len(majority_tags)}个标签")
        
        # 5. 找最低公共父节点
        common_parent = find_parent_with_multiple_descriptions(majority_tags)
        if not common_parent:
            await ctx.warning("未找到包含所有描述的公共父元素。")
            return False
        
        # 6. 遍历公共父节点的子节点，提取内容并写入JSON
        beautiful_soup = []
        for idx, child in enumerate(common_parent.children, start=1):
            if getattr(child, "prettify", None):
                content = child.prettify().strip()
                if content:
                    beautiful_soup.append({
                        "Order": idx,
                        "Content": content
                    })
        
        if beautiful_soup:
            out_path = THIS_DIR / "BeautifulSoup_Content.json"
            with open(out_path, "w", encoding="utf-8") as jf:
                json.dump(beautiful_soup, jf, ensure_ascii=False, indent=4)
            await ctx.info(f"已写入JSON: {out_path}")
            return True
        else:
            await ctx.warning("公共父节点下没有有效子节点内容。")
            return False
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        await ctx.error(f"处理标签定位时出错: {str(e)}")
        await ctx.error(f"错误详情: {error_trace}")
        return False

async def process_name_tag_location(ctx, file_path: str | None = None) -> bool:
    """
    处理商品名称标签定位
    """
    await ctx.info("开始处理商品名称标签定位...")
    
    # 从item_info.json加载商品名称
    try:
        product_names = await load_item_info(ctx, key='item')
        
        if not product_names:
            await ctx.error("未找到商品名称信息，请先运行OCR名称提取")
            return False
        
        # 执行标签定位处理
        try:
            result = await process_tag_location(ctx, product_names, file_path)
            if result:
                await ctx.info("商品名称标签定位处理完成")
            else:
                await ctx.warning("商品名称标签定位处理失败")
            return result
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            await ctx.error(f"商品名称标签定位处理出错: {str(e)}")
            await ctx.error(f"错误详情: {error_trace}")
            return False
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        await ctx.error(f"加载商品名称信息时出错: {str(e)}")
        await ctx.error(f"错误详情: {error_trace}")
        return False

async def process_price_tag_location(ctx, file_path: str | None = None) -> bool:
    """
    处理商品价格标签定位
    """
    await ctx.info("开始处理商品价格标签定位...")
    
    # 从item_info.json加载价格信息
    try:
        price_info = await load_item_info(ctx, key='price')
        
        if not price_info:
            await ctx.error("未找到价格信息，请先运行OCR价格提取")
            return False
        
        # 执行标签定位处理
        try:
            result = await process_tag_location(ctx, price_info, file_path)
            if result:
                await ctx.info("商品价格标签定位处理完成")
            else:
                await ctx.warning("商品价格标签定位处理失败")
            return result
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            await ctx.error(f"商品价格标签定位处理出错: {str(e)}")
            await ctx.error(f"错误详情: {error_trace}")
            return False
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        await ctx.error(f"加载价格信息时出错: {str(e)}")
        await ctx.error(f"错误详情: {error_trace}")
        return False

# ────────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────────────────────────
class CliContext:
    """命令行工具的上下文对象，模拟MCP Context接口"""
    async def info(self, msg):
        print(f"[INFO] {msg}")
    
    async def warning(self, msg):
        print(f"[WARNING] {msg}")
    
    async def error(self, msg):
        print(f"[ERROR] {msg}")

async def main_async():
    parser = argparse.ArgumentParser(description="标签定位工具")
    parser.add_argument("--type", choices=["name", "price"], default="name", 
                        help="处理类型: name(商品名称)或price(价格)")
    parser.add_argument("--filepath", default=None,
                        help="要处理的MHTML文件路径（可选，默认为mhtml_output下最新文件）")
    args = parser.parse_args()
    
    ctx = CliContext()
    
    if args.type == "name":
        await process_name_tag_location(ctx, args.filepath)
    else:
        await process_price_tag_location(ctx, args.filepath)

if __name__ == "__main__":
    asyncio.run(main_async())
