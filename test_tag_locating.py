"""
测试程序 - Tag_Locating.py 的独立测试版本
这个程序移除了MCP相关依赖，可以直接使用Python运行
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
import re
import sys
import time
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright, Page

# ────────────────────────────────────────────────────────────────────────────────
# 编码兼容：解决 Windows 下中文输出乱码
# ────────────────────────────────────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────────────
# 目录常量：基于脚本位置定位项目根和 mhtml 输出目录
# ────────────────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent             # 项目根目录
PROJECT_ROOT = THIS_DIR                                # 项目根目录
MHTML_DIR = PROJECT_ROOT / "mhtml_output"              # mhtml_output
EXTRACTOR_DIR = PROJECT_ROOT / "extractor"             # extractor

# ============================================================================
# 🔑  辅助工具
# ============================================================================

def _escape_regex(text: str) -> str:
    """对正则元字符转义"""
    return re.escape(text)


def _similar_ratio(a: str, b: str) -> float:
    """大小写无关的 SequenceMatcher 相似度 [0‑1]"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_dom_path(tag: Tag) -> str:
    """获取从当前标签到根节点的 DOM 路径（tagName 串联）。"""
    segments = []
    while tag is not None:
        segments.append(tag.name)
        tag = tag.parent  # type: ignore[attr‑defined]
    return " > ".join(reversed(segments))


# ----------------------------------------------------------------------------
# 核心 ✨ get_item_paths ✨：三段式定位策略实现
# ----------------------------------------------------------------------------

def get_item_paths(soup: BeautifulSoup, product_names: List[str]) -> Dict[str, List[Tag]]:
    """根据 *product_names* 在 DOM 中定位标签（或其公共父元素）的路径。

    **实现逻辑**
    1. 尝试 *精确/包含* 匹配整个字符串；成功则记录。
    2. 若失败 → **模糊**：使用相似度 > 0.65 的元素做粗定位（粗容器）。
    3. 在粗容器内部 **标签级分词**：按元素边界切词；对齐原字符串切分 token。
    4. 对每个 token 再做 **逐词精确定位**；找最低公共父元素作为最终标签。
    5. Fallback：仍无法定位，则拆词后直接在文档级搜索，每词独立处理。
    """

    paths: Dict[str, List[Tag]] = defaultdict(list)

    # 预编译常用函数
    def exact_or_contains(txt: str) -> Optional[Tag]:
        # 完整匹配
        exact = soup.find(lambda t: t.string and t.string.strip().lower() == txt.lower())
        if exact:
            return exact
        # 子串包含
        return soup.find(lambda t: t.string and txt.lower() in t.string.lower())

    for raw in product_names:
        raw_clean = raw.strip()
        if not raw_clean:
            continue

        # ——— ① 精确 / 包含匹配 ———
        tag = exact_or_contains(raw_clean)
        if tag:
            paths[get_dom_path(tag)].append(tag)
            continue  # ✅ 直接找到，跳过后续

        # ——— ② 模糊定位：找相似度最高的元素作为"粗容器" ———
        # 先粗暴拿所有包含单词的元素（防止全局遍历耗时）
        word_pat = re.compile(_escape_regex(raw_clean.split()[0]), re.I)
        candidates = [t for t in soup.find_all(string=word_pat) if isinstance(t, str)]
        best_container: Optional[Tag] = None
        best_score = 0.0
        for text_node in candidates:
            parent_el = text_node.parent  # type: ignore[assignment]
            text_val = text_node.strip()
            score = _similar_ratio(text_val, raw_clean)
            if score > best_score:
                best_score, best_container = score, parent_el
        if best_container is None or best_score < 0.65:
            # 进入 fallback：按 token 在全局搜索
            _record_by_tokens(soup, raw_clean, paths)
            continue

        # ——— ③ 标签级分词：在 best_container 内部按元素边界切词 ———
        tokens = _tokenize(raw_clean)
        token_tags = _locate_tokens_inside_container(best_container, tokens)
        if not token_tags:  # 没有全部 token => Fallback
            _record_by_tokens(soup, raw_clean, paths)
            continue

        # ——— ④ 找公共父元素作为最终定位 ———
        common_parent = _lowest_common_parent(token_tags)
        target_tag = common_parent if common_parent else best_container
        paths[get_dom_path(target_tag)].append(target_tag)

    return paths


# ----------------------------------------------------------------------------
#  辅助实现（标签级分词 & 公共父元素）
# ----------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """简单按非字母数字分词，过滤空 token。"""
    return [tok for tok in re.split(r"\W+", text) if tok]


def _locate_tokens_inside_container(container: Tag, tokens: List[str]) -> List[Tag]:
    """在 *container* 内逐词定位，要求每个 token 都能匹配到独立标签。"""

    def match_tag(root: Tag, token: str) -> Optional[Tag]:
        # 优先找文本完全等于 token 的元素；其次子串包含。
        exact = root.find(lambda t: t.string and t.string.strip().lower() == token.lower())
        if exact:
            return exact
        return root.find(lambda t: t.string and token.lower() in t.string.lower())

    matches: List[Tag] = []
    for tk in tokens:
        mt = match_tag(container, tk)
        if not mt:
            return []  # 有 token 未命中，则认为失败
        matches.append(mt)
    return matches


def _lowest_common_parent(tags: List[Tag]) -> Optional[Tag]:
    """返回一组标签的最低公共父元素。若不存在则返回 None。"""
    if not tags:
        return None
    # 先把各自祖先路径列出来（含自身）
    paths = []
    for t in tags:
        p: List[Tag] = []
        cur: Optional[Tag] = t
        while cur is not None:
            p.append(cur)
            cur = cur.parent  # type: ignore[assignment]
        paths.append(list(reversed(p)))

    # 对比公共前缀
    lcp: List[Tag] = []
    for zipped in zip(*paths):
        if all(node is zipped[0] for node in zipped):
            lcp.append(zipped[0])
        else:
            break
    return lcp[-1] if lcp else None


def _record_by_tokens(soup: BeautifulSoup, raw_clean: str, paths: Dict[str, List[Tag]]):
    """Fallback 逻辑：把 raw 拆分 token 后在全局搜索并记录到 paths。"""
    for tk in _tokenize(raw_clean):
        tag = soup.find(lambda t: t.string and tk.lower() in t.string.lower())
        if tag:
            paths[get_dom_path(tag)].append(tag)


# ============================================================================
#  其他功能函数
# ============================================================================

def filter_paths(paths: Dict[str, List[Tag]]) -> List[Tag]:
    """筛选出现次数最多的路径对应的标签列表，随机返回至多两个。"""
    if not paths:
        return []
    max_occurrence = max(len(tags) for tags in paths.values())
    filtered = {p: tags for p, tags in paths.items() if len(tags) == max_occurrence}
    candidate_tags = next(iter(filtered.values()), [])
    
    # 优先选择相邻的标签，而不是完全随机
    if len(candidate_tags) <= 2:
        return candidate_tags
    
    # 选择DOM结构上相近的标签
    return select_nearby_tags(candidate_tags, 2)


def select_nearby_tags(tags: List[Tag], count: int) -> List[Tag]:
    """
    从tags列表中选择最多count个在DOM结构中彼此"相近"的标签。
    "相近"通过DOM路径的相似度来确定。
    """
    if len(tags) <= count:
        return tags
    
    # 如果只需要一个标签，随机返回一个
    if count == 1:
        return [random.choice(tags)]
    
    # 计算所有标签对之间的DOM路径相似度
    best_score = -1
    best_pair = None
    
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            tag1, tag2 = tags[i], tags[j]
            path1 = get_dom_path(tag1)
            path2 = get_dom_path(tag2)
            
            # 计算DOM路径的相似度
            similarity = _similar_ratio(path1, path2)
            
            if similarity > best_score:
                best_score = similarity
                best_pair = (tag1, tag2)
    
    # 如果找到了相似度最高的一对，返回它们
    if best_pair:
        return list(best_pair)
    
    # 如果没有明显的最佳对，随机选择
    return random.sample(tags, count)


def find_parent_with_multiple_descriptions(tags: List[Tag]) -> Optional[Tag]:
    """在候选标签中找到最低公共父元素，要求它的子节点中包含所有标签文本。"""
    if not tags:
        return None
    parents = [tag.parent for tag in tags]
    while True:
        if all(parents[0] is p for p in parents):
            parent = parents[0]
            texts = [t.get_text() for t in tags]
            if all(any(txt in desc.get_text() for desc in parent.find_all()) for txt in texts):
                return parent  # type: ignore[return‑value]
        parents = [p.parent or p for p in parents]
        if all(p.name == "html" for p in parents):  # type: ignore[union‑attr]
            return None


def get_mhtml_file(file_path: str | None = None) -> Path:
    """获取要处理的 MHTML 文件"""
    if file_path:
        fp = Path(file_path)
    else:
        fp = next(MHTML_DIR.glob("*.mhtml"), None)  # type: ignore[assignment]
    if not fp or not fp.exists():
        raise FileNotFoundError(f"Cannot find relevant MHTML file: {fp}")
    return fp


async def get_html_content(file_path: Path) -> str:
    """使用 Playwright 异步 API 加载 MHTML 文件并获取 HTML 内容"""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto(file_path.as_uri())
    await asyncio.sleep(5)  # 简易等待
    html_content = await page.content()
    await browser.close()
    await playwright.stop()
    return html_content


def save_beautiful_soup_content(beautiful_soup: List[Dict], output_dir: Path = None) -> bool:
    """
    保存提取的内容到JSON文件
    """
    if beautiful_soup:
        if output_dir is None:
            output_dir = EXTRACTOR_DIR
        out_path = output_dir / "BeautifulSoup_Content.json"
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(beautiful_soup, jf, ensure_ascii=False, indent=4)
        print(f"已写入 JSON：{out_path}")
        return True
    else:
        print("公共父节点下没有有效子节点内容。")
        return False


def load_item_info(key: str = 'item') -> List[str]:
    """
    从item_info.json加载信息
    key可以是'item'(商品名称)或'price'(价格)
    """
    product_names = []
    item_info_path = PROJECT_ROOT / 'item_info.json'
    
    if not item_info_path.exists():
        item_info_path = EXTRACTOR_DIR / 'item_info.json'
    
    if not item_info_path.exists():
        print(f"找不到item_info.json文件")
        return []
    
    try:
        with open(item_info_path, 'r', encoding='utf-8') as f:
            try:
                item_data = json.load(f)
                product_names = [str(item.get(key, '')) for item in item_data if key in item]
                print(f"找到{len(product_names)}个{key}信息")
                return product_names
            except json.JSONDecodeError as e:
                print(f"解析{item_info_path}失败: {str(e)}")
                return []
    except Exception as e:
        print(f"读取{item_info_path}失败: {str(e)}")
        return []


# ────────────────────────────────────────────────────────────────────────────────
# 主流程函数
# ────────────────────────────────────────────────────────────────────────────────

async def process_tag_location(product_names: List[str], file_path: str | None = None) -> bool:
    """
    通用的标签定位处理流程(异步版本)
    """
    try:
        # 1. 获取MHTML文件
        fp = get_mhtml_file(file_path)
        print(f"处理MHTML文件: {fp}")
        
        # 2. 用Playwright加载页面获取HTML内容
        html_content = await get_html_content(fp)
        
        # 3. 解析DOM，定位产品名称对应的标签路径
        soup = BeautifulSoup(html_content, "html.parser")
        paths = get_item_paths(soup, product_names)
        print(f"找到的路径数量: {len(paths)}")
        
        # 4. 筛选出现次数最多的标签
        majority_tags = filter_paths(paths)
        if not majority_tags:
            print("警告: 没有匹配到有效的标签，跳过后续处理。")
            return False
        print(f"选取了{len(majority_tags)}个标签")
        
        # 5. 找最低公共父节点
        common_parent = find_parent_with_multiple_descriptions(majority_tags)
        if not common_parent:
            print("警告: 未找到包含所有描述的公共父元素。")
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
            out_path = EXTRACTOR_DIR / "BeautifulSoup_Content.json"
            with open(out_path, "w", encoding="utf-8") as jf:
                json.dump(beautiful_soup, jf, ensure_ascii=False, indent=4)
            print(f"已写入JSON: {out_path}")
            return True
        else:
            print("警告: 公共父节点下没有有效子节点内容。")
            return False
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"错误: 处理标签定位时出错: {str(e)}")
        print(f"错误详情: {error_trace}")
        return False


async def process_name_tag_location(file_path: str | None = None) -> bool:
    """
    处理商品名称标签定位
    """
    print("开始处理商品名称标签定位...")
    
    # 从item_info.json加载商品名称
    try:
        product_names = load_item_info(key='item')
        
        if not product_names:
            print("错误: 未找到商品名称信息，请先运行OCR名称提取")
            return False
        
        # 执行标签定位处理
        try:
            result = await process_tag_location(product_names, file_path)
            if result:
                print("商品名称标签定位处理完成")
            else:
                print("商品名称标签定位处理失败")
            return result
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"错误: 商品名称标签定位处理出错: {str(e)}")
            print(f"错误详情: {error_trace}")
            return False
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"错误: 加载商品名称信息时出错: {str(e)}")
        print(f"错误详情: {error_trace}")
        return False


async def process_price_tag_location(file_path: str | None = None) -> bool:
    """
    处理商品价格标签定位
    """
    print("开始处理商品价格标签定位...")
    
    # 从item_info.json加载价格信息
    try:
        price_info = load_item_info(key='price')
        
        if not price_info:
            print("错误: 未找到价格信息，请先运行OCR价格提取")
            return False
        
        # 执行标签定位处理
        try:
            result = await process_tag_location(price_info, file_path)
            if result:
                print("商品价格标签定位处理完成")
            else:
                print("商品价格标签定位处理失败")
            return result
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"错误: 商品价格标签定位处理出错: {str(e)}")
            print(f"错误详情: {error_trace}")
            return False
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"错误: 加载价格信息时出错: {str(e)}")
        print(f"错误详情: {error_trace}")
        return False


# ────────────────────────────────────────────────────────────────────────────────
# 测试功能
# ────────────────────────────────────────────────────────────────────────────────

async def test_with_real_data():
    """使用真实数据测试核心功能"""
    print("=" * 60)
    print(f"测试模式：使用真实 MHTML 文件和 item_info.json")
    print("=" * 60)
    
    # 1. 从 item_info.json 读取商品名称
    print("1. 从 item_info.json 读取商品名称...")
    product_names = load_item_info(key='item')
    
    if not product_names:
        print("错误: 未找到商品名称信息")
        return False
    
    print(f"   成功读取 {len(product_names)} 个商品名称")
    # 显示前几个商品名称
    for i, name in enumerate(product_names[:5], 1):
        print(f"   {i}. {name}")
    if len(product_names) > 5:
        print(f"   ... 还有 {len(product_names) - 5} 个商品")
    
    # 2. 从 mhtml_output 读取 MHTML 文件
    print("\n2. 从 mhtml_output 读取 MHTML 文件...")
    try:
        mhtml_file = get_mhtml_file()
        print(f"   找到 MHTML 文件: {mhtml_file}")
    except FileNotFoundError as e:
        print(f"   错误: {e}")
        return False
    
    # 3. 使用 Playwright 获取 HTML 内容
    print("\n3. 使用 Playwright 加载页面获取 HTML 内容...")
    try:
        html_content = await get_html_content(mhtml_file)
        print(f"   成功获取 HTML 内容，长度: {len(html_content)} 字符")
    except Exception as e:
        print(f"   错误: 获取 HTML 内容失败: {e}")
        return False
    
    # 4. 解析 HTML 并执行标签匹配
    soup = BeautifulSoup(html_content, "html.parser")    
    # 测试核心功能
    print("\n4. 测试 get_item_paths 函数...")
    paths = get_item_paths(soup, product_names)
    print(f"   找到路径数量: {len(paths)}")
    for path, tags in paths.items():
        print(f"   路径: {path} -> {len(tags)} 个标签")
    
    print("\n5. 测试 filter_paths 函数...")
    majority_tags = filter_paths(paths)
    print(f"   筛选出 {len(majority_tags)} 个主要标签")
    
    if not majority_tags:
        print("   警告: 没有匹配到有效的标签")
        return False
    
    print("\n6. 测试 find_parent_with_multiple_descriptions 函数...")
    common_parent = find_parent_with_multiple_descriptions(majority_tags)
    if common_parent:
        print(f"   找到公共父元素: {common_parent.name}")
        print(f"   父元素内容预览: {common_parent.get_text()[:100]}...")
    else:
        print("   未找到公共父元素")
        return False
    
    print("\n7. 提取并保存内容到 test_output.json...")
    beautiful_soup = []
    for idx, child in enumerate(common_parent.children, start=1):
        if getattr(child, "prettify", None):
            content = child.prettify().strip()
            if content:
                beautiful_soup.append({
                    "Order": idx,
                    "Content": content
                })
    
    print(f"   提取出 {len(beautiful_soup)} 个子元素")
    
    # 保存测试结果到 test_output.json
    test_output_path = PROJECT_ROOT / "test_output.json"
    with open(test_output_path, "w", encoding="utf-8") as f:
        json.dump(beautiful_soup, f, ensure_ascii=False, indent=4)
    print(f"   测试结果已保存到: {test_output_path}")
    
    print("\n测试完成！标签匹配成功，结果已保存到 test_output.json")
    return True


def test_with_sample_data():
    """使用真实数据测试，从mhtml_output读取文件，处理item_info.json数据，输出到BeautifulSoup_Content.json"""
    print("=" * 60)
    print("测试模式：使用真实数据")
    print("=" * 60)
    
    # 1. 从 item_info.json 读取商品名称
    print("1. 从 item_info.json 读取商品名称...")
    product_names = load_item_info(key='item')
    
    if not product_names:
        print("错误: 未找到商品名称信息")
        return False
    
    print(f"   成功读取 {len(product_names)} 个商品名称")
    for i, name in enumerate(product_names[:3], 1):
        print(f"   {i}. {name}")
    if len(product_names) > 3:
        print(f"   ... 还有 {len(product_names) - 3} 个商品")
    
    # 2. 从 mhtml_output 读取 MHTML 文件
    print("\n2. 从 mhtml_output 读取 MHTML 文件...")
    try:
        mhtml_file = get_mhtml_file()
        print(f"   找到 MHTML 文件: {mhtml_file}")
    except FileNotFoundError as e:
        print(f"   错误: {e}")
        return False
    
    # 3. 使用 Playwright 获取 HTML 内容
    print("\n3. 使用 Playwright 加载页面获取 HTML 内容...")
    try:
        html_content = asyncio.get_event_loop().run_until_complete(get_html_content(mhtml_file))
        print(f"   成功获取 HTML 内容，长度: {len(html_content)} 字符")
    except Exception as e:
        print(f"   错误: 获取 HTML 内容失败: {e}")
        return False
    
    # 4. 解析 HTML 并执行标签匹配
    soup = BeautifulSoup(html_content, "html.parser")    
    print("\n4. 执行标签匹配...")
    paths = get_item_paths(soup, product_names)
    print(f"   找到路径数量: {len(paths)}")
    
    # 5. 筛选出主要标签
    majority_tags = filter_paths(paths)
    print(f"   筛选出 {len(majority_tags)} 个主要标签")
    
    if not majority_tags:
        print("   警告: 没有匹配到有效的标签")
        return False
    
    # 6. 找到公共父元素
    common_parent = find_parent_with_multiple_descriptions(majority_tags)
    if common_parent:
        print(f"   找到公共父元素: {common_parent.name}")
    else:
        print("   未找到公共父元素")
        return False
    
    # 7. 提取并保存内容到 BeautifulSoup_Content.json
    beautiful_soup = []
    for idx, child in enumerate(common_parent.children, start=1):
        if getattr(child, "prettify", None):
            content = child.prettify().strip()
            if content:
                beautiful_soup.append({
                    "Order": idx,
                    "Content": content
                })
    
    print(f"   提取出 {len(beautiful_soup)} 个子元素")
    
    # 保存到EXTRACTOR_DIR/BeautifulSoup_Content.json
    output_path = EXTRACTOR_DIR / "BeautifulSoup_Content.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(beautiful_soup, f, ensure_ascii=False, indent=4)
    print(f"   结果已保存到: {output_path}")
    
    print("\n处理完成！")
    return True


# ────────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────────────────────────────────────

async def main_async():
    parser = argparse.ArgumentParser(description="Tag locating test tool")
    parser.add_argument("--type", choices=["name", "price", "test", "real"], default="real", 
                        help="Processing type: name (product name), price, test (sample data test), or real (use real data)")
    parser.add_argument("--filepath", default=None,
                        help="Path to the MHTML file to process (optional, defaults to the latest file in mhtml_output)")
    args = parser.parse_args()
    
    if args.type == "test":
        test_with_sample_data()
    elif args.type == "real":
        await test_with_real_data()
    elif args.type == "name":
        await process_name_tag_location(args.filepath)
    else:
        await process_price_tag_location(args.filepath)


def main():
    """同步入口函数"""
    print("Tag Locating 测试程序")
    print("使用方法:")
    print("  python test_tag_locating.py --type real     # 使用真实数据测试 (默认)")
    print("  python test_tag_locating.py --type test     # 运行示例数据测试")
    print("  python test_tag_locating.py --type name     # 处理商品名称定位")
    print("  python test_tag_locating.py --type price    # 处理商品价格定位")
    print("  python test_tag_locating.py --type real --filepath path/to/file.mhtml")
    print("")
    
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
