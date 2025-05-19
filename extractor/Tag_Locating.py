# extractor/Tag_Locating_2.py

import json
import os
import time
import random
import argparse
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

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

# ────────────────────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────────────────────
def main(product_names: list[str], file_path: str | None = None):
    # 1. 确定要处理的 MHTML 文件
    if file_path:
        fp = Path(file_path)
    else:
        # 找项目根的 mhtml_output 下最新的 .mhtml
        fp = next(MHTML_DIR.glob("*.mhtml"), None)
    if not fp or not fp.exists():
        raise FileNotFoundError(f"找不到要处理的 MHTML 文件：{fp}")

    # 2. 用 Playwright 加载页面
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 构造 file:// URL
        file_url = fp.as_uri()
        page.goto(file_url)

        # 等待页面加载（可根据实际情况改成 wait_for_selector）
        time.sleep(5)
        html_content = page.content()
        browser.close()

    # 3. 解析 DOM，定位产品名称对应的标签路径
    soup = BeautifulSoup(html_content, "html.parser")
    paths = get_item_paths(soup, product_names)
    print(f"找到的路径: {paths}")

    # 4. 筛选出现次数最多的标签
    majority_tags = filter_paths(paths)
    if not majority_tags:
        print("没有匹配到有效的标签，跳过后续处理。")
        return
    print(f"选取的标签: {majority_tags}")

    # 5. 找最低公共父节点
    common_parent = find_parent_with_multiple_descriptions(majority_tags)
    if not common_parent:
        print("未找到包含所有描述的公共父元素。")
        return

    # 6. 遍历公共父节点的子节点，提取内容并写入 JSON
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
        print(f"已写入 JSON：{out_path}")
    else:
        print("公共父节点下没有有效子节点内容。")

# ────────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tag_Locating_2 on an MHTML file")
    parser.add_argument("filepath", nargs="?", default=None,
                        help="要处理的 MHTML 文件路径（可选，默认为 mhtml_output 下最新文件）")
    args = parser.parse_args()

    # 从项目根或 extractor 目录读取 item_info.json，提取 product names
    product_names = []
    item_info_path = 'item_info.json'
    if os.path.exists(item_info_path):
        with open(item_info_path, 'r', encoding='utf-8') as f:
            item_data = json.load(f)
            product_names = [str(item['price']) for item in item_data]
            print("Product names:", product_names)
    else:
        print(f"{item_info_path} 不存在，prices 为空。")

    main(product_names, args.filepath)
