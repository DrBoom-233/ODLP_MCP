"""
标签定位模块 - 提供从 MHTML 文件中定位商品名称和价格标签的功能。

重写要点（2025‑06‑17）：
1. **新增三段式定位策略**：模糊定位 ➜ 标签级分词 ➜ 逐词精确定位。
2. **核心逻辑全部封装在 `get_item_paths`**，对外函数签名不变，Server 侧无需改动。
3. 仍保留原有 BeautifulSoup Fallback，保证在极端结构下也能产出结果。
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
from playwright.sync_api import sync_playwright  # 保留同步版接口，部分 CLI 调用仍依赖

# ────────────────────────────────────────────────────────────────────────────────
# 编码兼容：解决 Windows 下中文输出乱码
# ────────────────────────────────────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────────────
# 目录常量：基于脚本位置定位项目根和 mhtml 输出目录
# ────────────────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent             # extractor/
PROJECT_ROOT = THIS_DIR.parent                         # mcp‑project 根目录
MHTML_DIR = PROJECT_ROOT / "mhtml_output"              # mhtml_output 与 extractor 同级

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

        # ——— ② 模糊定位：找相似度最高的元素作为“粗容器” ———
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


def _lowest_common_parent(tags: List[Tag], ctx=None) -> Optional[Tag]:
    """返回一组标签的最低公共父元素。若不存在则返回 None。"""
    if not tags:
        return None
        
    # 添加调试信息：输出参与查找的标签信息
    if ctx:
        ctx.info(f"\n===== 开始查找 {len(tags)} 个标签的最小公共父元素 =====")
        for i, tag in enumerate(tags, 1):
            ctx.info(f"标签 {i}: <{tag.name}> - 文本: {tag.get_text().strip()[:50]}")
            ctx.info(f"DOM路径: {get_dom_path(tag)}")
    
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
    
    result = lcp[-1] if lcp else None
  
    return result


def _record_by_tokens(soup: BeautifulSoup, raw_clean: str, paths: Dict[str, List[Tag]]):
    """Fallback 逻辑：把 raw 拆分 token 后在全局搜索并记录到 paths。"""
    for tk in _tokenize(raw_clean):
        tag = soup.find(lambda t: t.string and tk.lower() in t.string.lower())
        if tag:
            paths[get_dom_path(tag)].append(tag)


# ============================================================================
#  其余原有代码基本保持 **不变**
#  · get_mhtml_file
#  · get_html_content
#  · filter_paths, find_parent_with_multiple_descriptions
#  · process_* 系列接口
# ============================================================================

# 以下内容从旧实现拷贝，仅删去不必要 import，逻辑保持原状。

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

def filter_paths(paths: Dict[str, List[Tag]]) -> List[Tag]:
    """筛选出现次数最多的路径对应的标签列表，智能返回至多两个。"""
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


def find_parent_with_multiple_descriptions(tags: List[Tag], ctx=None) -> Optional[Tag]:
    """
    在候选标签中找到最低公共父元素（基于 DOM 路径），
    要求它的子节点中包含所有标签文本。
    """
    if not tags:
        return None
    
    # 保留输入标签信息的调试输出
    if ctx:
        ctx.info(f"\n===== 开始查找 {len(tags)} 个标签的最低公共父元素 =====")
        for i, tag in enumerate(tags, 1):
            ctx.info(f"输入标签 {i}: <{tag.name}> - 文本: {tag.get_text().strip()[:50]}")
            ctx.info(f"  DOM路径: {get_dom_path(tag)}")
    
    parents = [tag.parent for tag in tags]
    # if ctx:
    #     ctx.info(f"\n正在检查直接父元素...")
    #     for i, parent in enumerate(parents, 1):
    #         ctx.info(f"父元素 {i}: <{parent.name}> - 路径: {get_dom_path(parent)}")
    
    level = 1
    while True:
        # 改用 DOM 路径进行比较
        ref_path = get_dom_path(parents[0])
        if all(get_dom_path(p) == ref_path for p in parents):
            parent = parents[0]
            texts = [t.get_text() for t in tags]
            
            # if ctx:
            #     ctx.info(f"\n在第 {level} 层找到相同DOM路径: {ref_path}")
            #     ctx.info(f"正在检查是否包含所有标签文本...")
            
            if all(any(txt in desc.get_text() for desc in parent.find_all()) for txt in texts):
                if ctx:
                    ctx.info(f"✓ 找到包含所有文本的父元素: <{parent.name}> - 路径: {get_dom_path(parent)}")
                    ctx.info(f"  内容预览: {parent.get_text().strip()[:100]}...")
                return parent
            # elif ctx:
            #     ctx.info(f"✗ 父元素 <{parent.name}> 不包含所有标签文本，继续向上查找")
        
        # 继续往上找
        level += 1
        parents = [p.parent or p for p in parents]
        # if ctx:
        #     ctx.info(f"\n查找第 {level} 层父元素...")
        
        if all(p.name == "html" for p in parents):
            if ctx:
                ctx.info("已到达HTML根节点，未找到包含所有文本的父元素")
            return None


# get_mhtml_file, get_html_content, save_beautiful_soup_content, load_item_info,
# process_tag_location, process_name_tag_location, process_price_tag_location,
# CLI 部分均保持不变，直接从旧文件 copy 过来（略）。

from typing import Tuple  # 需要在后面继续使用

# —— 以下整段直接保留旧实现 ——

async def get_mhtml_file(file_path: str | None = None) -> Path:
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
async def process_tag_location(ctx, product_names: List[str], file_path: str | None = None, data_field: str = 'item') -> bool:
    """
    通用的标签定位处理流程(异步版本)
    
    Args:
        ctx: 上下文对象
        product_names: 产品名称列表（如果提供则直接使用，否则从JSON加载）
        file_path: MHTML文件路径
        data_field: 要从JSON加载的字段名，可以是 'item', 'price' 等
    """
    try:
        # 1. 获取MHTML文件
        await ctx.info(f"开始处理标签定位...")
        await ctx.info(f"工作目录: {Path.cwd()}")
        await ctx.info(f"MHTML目录: {MHTML_DIR}")
        
        fp = await get_mhtml_file(file_path)
        await ctx.info(f"处理MHTML文件: {fp}")
        
        # 2. 用Playwright加载页面获取HTML内容
        await ctx.info("开始使用Playwright加载页面...")
        html_content = await get_html_content(fp)
        soup = BeautifulSoup(html_content, "html.parser")
        await ctx.info(f"成功获取 HTML 内容，长度: {len(html_content)} 字符")
        
        # 3. 如果没有提供product_names，则从item_info.json加载指定字段
        all_items = product_names if product_names else []
        
        if not all_items:
            item_info_path = THIS_DIR / 'item_info.json'
            if item_info_path.exists():
                with open(item_info_path, 'r', encoding='utf-8') as f:
                    try:
                        item_data = json.load(f)
                        # 修改这里：使用传入的data_field参数
                        all_items = [str(item.get(data_field, '')) for item in item_data if data_field in item and item.get(data_field)]
                        await ctx.info(f"从item_info.json加载了{len(all_items)}个{data_field}字段的数据")
                    except json.JSONDecodeError:
                        await ctx.warning("解析item_info.json失败")
            else:
                await ctx.error(f"未提供数据且找不到item_info.json文件")
                return False
        
        # 确保有足够的item
        if len(all_items) < 2:
            await ctx.warning(f"只有{len(all_items)}个有效的{data_field}数据，至少需要2个")
            if not all_items:
                return False
        
        await ctx.info(f"将处理{len(all_items)}个{data_field}数据: {all_items[:5]}{'...' if len(all_items) > 5 else ''}")
        
        # ================== 新增：先尝试精确匹配 ==================
        await ctx.info("\n第一步：尝试精确匹配...")
        exact_match_tags = []
        
        # 对每个item尝试精确匹配
        for item in all_items:
            # 1. 首先尝试 string 精确匹配 (最严格，要求只有一个文本节点)
            exact_tag = soup.find(lambda t: t.string and t.string.strip().lower() == item.lower())
            if exact_tag:
                exact_match_tags.append(exact_tag)
                await ctx.info(f"  ✓ string精确匹配成功: '{item}' -> <{exact_tag.name}>")
                continue
                
            # 2. 然后尝试 get_text() 精确匹配 (处理有子元素的标签)
            exact_tag = soup.find(lambda t: t.get_text().strip().lower() == item.lower())
            if exact_tag:
                exact_match_tags.append(exact_tag)
                await ctx.info(f"  ✓ get_text精确匹配成功: '{item}' -> <{exact_tag.name}>")
                continue
            
            # 3. 最后尝试部分匹配 (最宽松)
            partial_tag = soup.find(lambda t: t.get_text() and item.lower() in t.get_text().strip().lower())
            if partial_tag:
                exact_match_tags.append(partial_tag)
                await ctx.info(f"  ✓ 部分匹配成功: '{item}' 在 <{partial_tag.name}> 中")
        
        # 如果精确匹配找到了足够的标签，直接使用
        if len(exact_match_tags) >= 2:
            await ctx.info(f"\n精确匹配成功找到 {len(exact_match_tags)} 个标签，跳过分词步骤")
            
            # 只随机选择最多两个标签参与最小公共父元素查找
            selected_tags = exact_match_tags
            if len(exact_match_tags) > 2:
                selected_tags = random.sample(exact_match_tags, 2)
                await ctx.info(f"从 {len(exact_match_tags)} 个精确匹配标签中随机选择 2 个用于查找公共父元素")
            
            # 保留输入标签信息的调试输出
            await ctx.info(f"\n===== 开始查找 {len(selected_tags)} 个标签的最低公共父元素 =====")
            for i, tag in enumerate(selected_tags, 1):
                await ctx.info(f"输入标签 {i}: <{tag.name}> - 文本: {tag.get_text().strip()[:50]}")
                await ctx.info(f"  DOM路径: {get_dom_path(tag)}")
                
            # 寻找这些标签的最小公共父元素
            common_parent = _lowest_common_parent(selected_tags)
            
            if not common_parent:
                await ctx.warning("未找到包含所有标签的公共父元素，尝试使用分词方法")
            elif common_parent.name in ['head', 'body', 'html']:
                await ctx.warning(f"精确匹配的公共父元素是 <{common_parent.name}>，结果太大，尝试使用分词方法")
            else:
                await ctx.info(f"精确匹配成功找到合适的公共父元素: <{common_parent.name}>")
                # 跳转到处理公共父元素部分，不再执行分词
                goto_process_common_parent = True
        else:
            await ctx.info(f"精确匹配只找到 {len(exact_match_tags)} 个标签，不够用，继续尝试分词匹配")
            goto_process_common_parent = False
        
        # ================== 如果精确匹配失败，再尝试分词匹配 ==================
        if not goto_process_common_parent:
            await ctx.info("\n第二步：尝试分词匹配...")
            # 4. 对所有item进行分词
            tokenized_items = [_tokenize(item) for item in all_items]
            
            # 找出最短的分词长度，确保位置对应
            min_token_length = min(len(tokens) for tokens in tokenized_items)
            await ctx.info(f"最短分词长度: {min_token_length}")
            
            # 5. 按位置尝试匹配，寻找DOM路径相似的标签
            best_position = None
            best_position_tags = []
            best_similarity_score = 0.0
            
            for pos in range(min_token_length):
                # 获取当前位置的所有分词
                current_tokens = [item_tokens[pos] for item_tokens in tokenized_items]
                await ctx.info(f"\n尝试位置{pos+1}的分词: {', '.join(current_tokens[:5])}{'...' if len(current_tokens) > 5 else ''}")
                
                # 查找匹配标签
                position_tags = []
                for i, token in enumerate(current_tokens):
                    tag = soup.find(lambda t: t.string and token.lower() in t.string.lower())
                    if tag:
                        position_tags.append(tag)
                
                await ctx.info(f"  位置{pos+1}找到{len(position_tags)}/{len(current_tokens)}个匹配标签")
                
                # 只有找到足够多的标签才进行DOM路径比较
                if len(position_tags) >= 2:
                    # 计算DOM路径的相似度
                    dom_paths = [get_dom_path(tag) for tag in position_tags]
                    
                    # 计算平均相似度
                    path_similarities = []
                    for i in range(len(dom_paths)):
                        for j in range(i+1, len(dom_paths)):
                            similarity = _similar_ratio(dom_paths[i], dom_paths[j])
                            path_similarities.append(similarity)
                    
                    avg_similarity = sum(path_similarities) / len(path_similarities) if path_similarities else 0
                    await ctx.info(f"  DOM路径平均相似度: {avg_similarity:.4f}")
                    
                    # 记录最佳位置
                    if avg_similarity > best_similarity_score:
                        best_similarity_score = avg_similarity
                        best_position = pos
                        best_position_tags = position_tags
                        await ctx.info(f"  ✓ 更新最佳位置为位置{pos+1}，相似度{avg_similarity:.4f}")
            
            # 6. 如果找到了最佳位置，使用该位置的标签
            if best_position is not None and best_similarity_score > 0.6:  # 设置一个相似度阈值
                await ctx.info(f"\n使用最佳位置{best_position+1}的标签，DOM路径相似度: {best_similarity_score:.4f}")
                # 显示找到的标签
                for i, tag in enumerate(best_position_tags[:5]):
                    await ctx.info(f"  标签{i+1}: <{tag.name}> - {tag.get_text().strip()[:50]}")
                    await ctx.info(f"  DOM路径: {get_dom_path(tag)}")
                
                # 只随机选择最多两个标签参与最小公共父元素查找
                selected_tags = best_position_tags
                if len(best_position_tags) > 2:
                    selected_tags = random.sample(best_position_tags, 2)
                    await ctx.info(f"\n从 {len(best_position_tags)} 个标签中随机选择 2 个用于查找公共父元素")
                
                # 保留输入标签信息的调试输出
                await ctx.info(f"\n===== 开始查找 {len(selected_tags)} 个标签的最低公共父元素 =====")
                for i, tag in enumerate(selected_tags, 1):
                    await ctx.info(f"输入标签 {i}: <{tag.name}> - 文本: {tag.get_text().strip()[:50]}")
                    await ctx.info(f"  DOM路径: {get_dom_path(tag)}")
                    
                # 寻找这些标签的最小公共父元素
                common_parent = _lowest_common_parent(selected_tags)
                
                if not common_parent:
                    await ctx.warning("未找到包含所有标签的公共父元素")
                    return False
                
                await ctx.info(f"找到最小公共父元素: <{common_parent.name}>")
                
                # 如果公共父元素是head或body，直接报错而不是使用备选
                if common_parent.name in ['head', 'body', 'html']:
                    await ctx.error(f"公共父元素是 <{common_parent.name}>，匹配结果太大，定位失败")
                    return False
            else:
                await ctx.warning("未找到DOM路径相似度足够高的位置")
                # 使用传统方法
                await ctx.info("退回到传统方法，使用所有分词进行匹配")
                
                # 使用get_item_paths函数处理前几个商品名称
                sample_names = all_items[:3] if len(all_items) >= 3 else all_items
                paths = get_item_paths(soup, sample_names)
                await ctx.info(f"找到的匹配项数量: {len(paths)}")
                
                # 筛选出现次数最多的标签
                majority_tags = filter_paths(paths)
                if not majority_tags:
                    await ctx.warning("没有匹配到有效的标签，跳过后续处理。")
                    return False
                
                await ctx.info(f"选取了{len(majority_tags)}个最佳匹配标签")
                
                # 只随机选择最多两个标签参与最小公共父元素查找
                selected_tags = majority_tags
                if len(majority_tags) > 2:
                    selected_tags = random.sample(majority_tags, 2)
                    await ctx.info(f"从 {len(majority_tags)} 个标签中随机选择 2 个用于查找公共父元素")
                
                # 找最低公共父节点
                common_parent = find_parent_with_multiple_descriptions(selected_tags)
                if not common_parent:
                    await ctx.warning("未找到包含所有描述的公共父元素。")
                    return False

                # 如果公共父元素是head或body，直接报错
                if common_parent.name in ['head', 'body', 'html']:
                    await ctx.error(f"公共父元素是 <{common_parent.name}>，匹配结果太大，定位失败")
                    return False
        
        # ================== 处理找到的公共父元素 ==================
        await ctx.info(f"\n最终使用的父元素: <{common_parent.name}> - 内容预览: {common_parent.get_text().strip()[:100]}...")
        
        # 7. 遍历公共父节点的子节点，提取内容并写入JSON
        beautiful_soup = []
        child_count = 0
        for idx, child in enumerate(common_parent.children, start=1):
            child_count += 1
            if getattr(child, "prettify", None):
                content = child.prettify().strip()
                if content:
                    beautiful_soup.append({
                        "Order": idx,
                        "Content": content
                    })
        
        await ctx.info(f"公共父元素总共有 {child_count} 个子元素")
        await ctx.info(f"提取出 {len(beautiful_soup)} 个有效子元素")
        
        if beautiful_soup:
            out_path = THIS_DIR / "BeautifulSoup_Content.json"
            with open(out_path, "w", encoding="utf-8") as jf:
                json.dump(beautiful_soup, jf, ensure_ascii=False, indent=4)
            await ctx.info(f"已写入JSON: {out_path}")
            return True
        else:
            await ctx.warning("公共父节点下没有有效子节点内容。")
            await ctx.warning("可能的原因：")
            await ctx.warning("1. 公共父元素为空或只包含文本节点")
            await ctx.warning("2. 子元素无法prettify（可能是文本节点）")
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
        # 尝试先加载item字段，如果没有则尝试其他可能的字段
        product_names = await load_item_info(ctx, key='item')
        
        if not product_names:
            # 如果没有item字段，尝试使用price字段作为替代
            await ctx.warning("未找到item字段，尝试使用price字段")
            product_names = await load_item_info(ctx, key='price')
            
            if not product_names:
                await ctx.error("未找到item或price字段信息")
                return False
        
        # 执行标签定位处理，传入数据类型
        field_type = 'item' if 'item' in str(product_names[0]) else 'price'
        result = await process_tag_location(ctx, product_names, file_path, data_field=field_type)
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

async def process_price_tag_location(ctx, file_path: str | None = None) -> bool:
    """
    处理商品价格标签定位
    """
    await ctx.info("开始处理商品价格标签定位...")
    
    # 从item_info.json加载价格信息
    try:
        price_info = await load_item_info(ctx, key='price')
        
        if not price_info:
            await ctx.error("未找到price字段信息")
            return False
        
        # 执行标签定位处理
        result = await process_tag_location(ctx, price_info, file_path, data_field='price')
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
    parser = argparse.ArgumentParser(description="Tag locating tool")
    parser.add_argument("--type", choices=["name", "price"], default="name", 
                        help="Processing type: name (product name) or price")
    parser.add_argument("--filepath", default=None,
                        help="Path to the MHTML file to process (optional, defaults to the latest file in mhtml_output)")
    args = parser.parse_args()
    
    ctx = CliContext()
    
    if args.type == "name":
        await process_name_tag_location(ctx, args.filepath)
    else:
        await process_price_tag_location(ctx, args.filepath)

if __name__ == "__main__":
    asyncio.run(main_async())
