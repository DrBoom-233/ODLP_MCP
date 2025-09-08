"""
Tag Locating Module - Provides functionality to locate product name and price tags from MHTML files.

Rewrite Highlights (2025‑06‑17):
1. **Added Three-Stage Locating Strategy**: Fuzzy locating ➜ Tag-level tokenization ➜ Precise word-by-word locating.
2. **Core logic encapsulated in `get_item_paths`**, external function signature remains unchanged, no changes needed on the Server side.
3. Retains the original BeautifulSoup Fallback to ensure results under extreme structures.
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
# Encoding Compatibility: Resolves garbled Chinese output on Windows
# ────────────────────────────────────────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────────────
# Directory Constants: Locate project root and mhtml output directory based on script location
# ────────────────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent             # extractor/
PROJECT_ROOT = THIS_DIR.parent                         # mcp‑project 根目录
MHTML_DIR = PROJECT_ROOT / "mhtml_output"              # mhtml_output 与 extractor 同级

# ============================================================================
# 🔑  Helper Utilities
# ============================================================================

def _escape_regex(text: str) -> str:
    """Escape regex metacharacters"""
    return re.escape(text)


def _similar_ratio(a: str, b: str) -> float:
    """Case-insensitive SequenceMatcher similarity [0-1]"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_dom_path(tag: Tag) -> str:
    """Get the DOM path (concatenated tagNames) from the current tag to the root node."""
    segments = []
    while tag is not None:
        segments.append(tag.name)
        tag = tag.parent  # type: ignore[attr-defined]
    return " > ".join(reversed(segments))



# ----------------------------------------------------------------------------
# Core ✨ get_item_paths ✨: Three-stage locating strategy implementation
# ----------------------------------------------------------------------------

def get_item_paths(soup: BeautifulSoup, product_names: List[str]) -> Dict[str, List[Tag]]:
    """Locate tags (or their common parent elements) in the DOM based on *product_names*.

    **Implementation Logic**
    1. Try *exact/contains* match of the full string; record if successful.
    2. If failed → **Fuzzy**: use elements with similarity > 0.65 as coarse containers.
    3. Inside the coarse container, perform **tag-level tokenization**: split by element boundaries and align with tokens from the original string.
    4. For each token, perform **word-by-word exact locating**; find the lowest common parent element as the final tag.
    5. Fallback: If still not located, split into tokens and search globally, processing each token independently.
    """

    paths: Dict[str, List[Tag]] = defaultdict(list)

    # Pre-compile commonly used function
    def exact_or_contains(txt: str) -> Optional[Tag]:
        # Exact match
        exact = soup.find(lambda t: t.string and t.string.strip().lower() == txt.lower())
        if exact:
            return exact
        # Substring match
        return soup.find(lambda t: t.string and txt.lower() in t.string.lower())

    for raw in product_names:
        raw_clean = raw.strip()
        if not raw_clean:
            continue

        # ——— ① Exact / contains match ———
        tag = exact_or_contains(raw_clean)
        if tag:
            paths[get_dom_path(tag)].append(tag)
            continue  # ✅ Found directly, skip following

        # ——— ② Fuzzy locating: find the element with the highest similarity as "coarse container" ———
        # First roughly collect all elements containing the first word (to avoid costly global traversal)
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
            # Enter fallback: search globally by tokens
            _record_by_tokens(soup, raw_clean, paths)
            continue

        # ——— ③ Tag-level tokenization: split inside best_container by element boundaries ———
        tokens = _tokenize(raw_clean)
        token_tags = _locate_tokens_inside_container(best_container, tokens)
        if not token_tags:  # Not all tokens matched => Fallback
            _record_by_tokens(soup, raw_clean, paths)
            continue

        # ——— ④ Find the common parent element as the final locating ———
        common_parent = _lowest_common_parent(token_tags)
        target_tag = common_parent if common_parent else best_container
        paths[get_dom_path(target_tag)].append(target_tag)

    return paths


# ----------------------------------------------------------------------------
#  Helper implementations (tag-level tokenization & common parent element)
# ----------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple tokenization by non-alphanumeric characters, filter out empty tokens."""
    return [tok for tok in re.split(r"\W+", text) if tok]


def _locate_tokens_inside_container(container: Tag, tokens: List[str]) -> List[Tag]:
    """Locate tokens one by one inside *container*, requiring each token to match an independent tag."""

    def match_tag(root: Tag, token: str) -> Optional[Tag]:
        # Prefer exact text match; fallback to substring match.
        exact = root.find(lambda t: t.string and t.string.strip().lower() == token.lower())
        if exact:
            return exact
        return root.find(lambda t: t.string and token.lower() in t.string.lower())

    matches: List[Tag] = []
    for tk in tokens:
        mt = match_tag(container, tk)
        if not mt:
            return []  # If any token is not matched, consider it a failure
        matches.append(mt)
    return matches


def _lowest_common_parent(tags: List[Tag], ctx=None) -> Optional[Tag]:
    """Return the lowest common parent of a set of tags. Return None if it does not exist."""
    if not tags:
        return None
        
    # Add debug information: output the tags being checked
    if ctx:
        ctx.info(f"\n===== Start finding lowest common parent of {len(tags)} tags =====")
        for i, tag in enumerate(tags, 1):
            ctx.info(f"Tag {i}: <{tag.name}> - Text: {tag.get_text().strip()[:50]}")
            ctx.info(f"DOM Path: {get_dom_path(tag)}")
    
    # Build ancestor paths for each tag (including itself)
    paths = []
    for t in tags:
        p: List[Tag] = []
        cur: Optional[Tag] = t
        while cur is not None:
            p.append(cur)
            cur = cur.parent  # type: ignore[assignment]
        paths.append(list(reversed(p)))

    # Compare common prefix
    lcp: List[Tag] = []
    for zipped in zip(*paths):
        if all(node is zipped[0] for node in zipped):
            lcp.append(zipped[0])
        else:
            break
    
    result = lcp[-1] if lcp else None
  
    return result


def _record_by_tokens(soup: BeautifulSoup, raw_clean: str, paths: Dict[str, List[Tag]]):
    """Fallback logic: split raw string into tokens, search globally, and record into paths."""
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
    Select up to *count* tags from the list that are "close" to each other in the DOM structure.
    "Closeness" is determined by the similarity of their DOM paths.
    """
    if len(tags) <= count:
        return tags
    
    # If only one tag is needed, return one randomly
    if count == 1:
        return [random.choice(tags)]
    
    # Calculate DOM path similarity for all tag pairs
    best_score = -1
    best_pair = None
    
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            tag1, tag2 = tags[i], tags[j]
            path1 = get_dom_path(tag1)
            path2 = get_dom_path(tag2)
            
            # Compute DOM path similarity
            similarity = _similar_ratio(path1, path2)
            
            if similarity > best_score:
                best_score = similarity
                best_pair = (tag1, tag2)
    
    # If a pair with the highest similarity is found, return them
    if best_pair:
        return list(best_pair)
    
    # If no clear best pair, return randomly selected ones
    return random.sample(tags, count)


def filter_paths(paths: Dict[str, List[Tag]]) -> List[Tag]:
    """Filter the tag list corresponding to the most frequent paths, intelligently returning up to two."""
    if not paths:
        return []
    max_occurrence = max(len(tags) for tags in paths.values())
    filtered = {p: tags for p, tags in paths.items() if len(tags) == max_occurrence}
    candidate_tags = next(iter(filtered.values()), [])
    
    # Prefer choosing adjacent tags instead of completely random
    if len(candidate_tags) <= 2:
        return candidate_tags
    
    # Choose tags that are close in DOM structure
    return select_nearby_tags(candidate_tags, 2)


def find_parent_with_multiple_descriptions(tags: List[Tag], ctx=None) -> Optional[Tag]:
    """
    Find the lowest common parent element among candidate tags (based on DOM path),
    requiring that its child nodes contain all tag texts.
    """
    if not tags:
        return None
    
    # Keep debug output of input tag information
    if ctx:
        ctx.info(f"\n===== Start finding lowest common parent of {len(tags)} tags =====")
        for i, tag in enumerate(tags, 1):
            ctx.info(f"Input Tag {i}: <{tag.name}> - Text: {tag.get_text().strip()[:50]}")
            ctx.info(f"  DOM Path: {get_dom_path(tag)}")
    
    parents = [tag.parent for tag in tags]
    # if ctx:
    #     ctx.info(f"\nChecking direct parent elements...")
    #     for i, parent in enumerate(parents, 1):
    #         ctx.info(f"Parent {i}: <{parent.name}> - Path: {get_dom_path(parent)}")
    
    level = 1
    while True:
        # Compare using DOM path
        ref_path = get_dom_path(parents[0])
        if all(get_dom_path(p) == ref_path for p in parents):
            parent = parents[0]
            texts = [t.get_text() for t in tags]
            
            # if ctx:
            #     ctx.info(f"\nAt level {level}, found identical DOM path: {ref_path}")
            #     ctx.info(f"Checking whether it contains all tag texts...")
            
            if all(any(txt in desc.get_text() for desc in parent.find_all()) for txt in texts):
                if ctx:
                    ctx.info(f"✓ Found a parent element containing all texts: <{parent.name}> - Path: {get_dom_path(parent)}")
                    ctx.info(f"  Content Preview: {parent.get_text().strip()[:100]}...")
                return parent
            # elif ctx:
            #     ctx.info(f"✗ Parent <{parent.name}> does not contain all texts, continue searching upward")
        
        # Continue searching upward
        level += 1
        parents = [p.parent or p for p in parents]
        # if ctx:
        #     ctx.info(f"\nSearching level {level} parent elements...")
        
        if all(p.name == "html" for p in parents):
            if ctx:
                ctx.info("Reached HTML root node, no parent element containing all texts found")
            return None


# get_mhtml_file, get_html_content, save_beautiful_soup_content, load_item_info,
# process_tag_location, process_name_tag_location, process_price_tag_location,
# CLI 部分均保持不变，直接从旧文件 copy 过来（略）。

from typing import Tuple  # 需要在后面继续使用

# —— 以下整段直接保留旧实现 ——

async def get_mhtml_file(file_path: str | None = None) -> Path:
    """Get the MHTML file to be processed"""
    if file_path:
        fp = Path(file_path)
    else:
        fp = next(MHTML_DIR.glob("*.mhtml"), None)  # type: ignore[assignment]
    if not fp or not fp.exists():
        raise FileNotFoundError(f"Cannot find relevant MHTML file: {fp}")
    return fp


async def get_html_content(file_path: Path) -> str:
    """Use Playwright async API to load the MHTML file and retrieve HTML content"""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto(file_path.as_uri())
    await asyncio.sleep(5)  # Simple wait
    html_content = await page.content()
    await browser.close()
    await playwright.stop()
    return html_content


def save_beautiful_soup_content(beautiful_soup: List[Dict]) -> bool:
    """
    Save the extracted content to a JSON file
    """
    if beautiful_soup:
        out_path = THIS_DIR / "BeautifulSoup_Content.json"
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(beautiful_soup, jf, ensure_ascii=False, indent=4)
        print(f"Written to JSON: {out_path}")
        return True
    else:
        print("No valid child node content under the common parent element.")
        return False


async def load_item_info(ctx, key: str = 'item') -> List[str]:
    """
    Load information from item_info.json
    The key can be 'item' (product name) or 'price' (price).
    """
    product_names = []
    item_info_path = THIS_DIR / 'item_info.json'
    
    if not item_info_path.exists():
        await ctx.error(f"Cannot find item_info.json file")
        return []
    
    try:
        with open(item_info_path, 'r', encoding='utf-8') as f:
            try:
                item_data = json.load(f)
                product_names = [str(item.get(key, '')) for item in item_data if key in item]
                await ctx.info(f"Found {len(product_names)} {key} entries")
                return product_names
            except json.JSONDecodeError as e:
                await ctx.error(f"Failed to parse {item_info_path}: {str(e)}")
                return []
    except Exception as e:
        await ctx.error(f"Failed to read {item_info_path}: {str(e)}")
        return []


# ────────────────────────────────────────────────────────────────────────────────
# Main process function
# ────────────────────────────────────────────────────────────────────────────────
async def process_tag_location(ctx, product_names: List[str], file_path: str | None = None, data_field: str = 'item') -> bool:
    """
    General tag locating process (asynchronous version)
    
    Args:
        ctx: Context object
        product_names: List of product names (use directly if provided, otherwise load from JSON)
        file_path: Path to the MHTML file
        data_field: Field name to load from JSON, can be 'item', 'price', etc.
    """
    try:
        # 1. Get MHTML file
        await ctx.info(f"Start processing tag location...")
        await ctx.info(f"Working directory: {Path.cwd()}")
        await ctx.info(f"MHTML directory: {MHTML_DIR}")
        
        fp = await get_mhtml_file(file_path)
        await ctx.info(f"Processing MHTML file: {fp}")
        
        # 2. Load HTML content using Playwright
        await ctx.info("Start loading page with Playwright...")
        html_content = await get_html_content(fp)
        soup = BeautifulSoup(html_content, "html.parser")
        await ctx.info(f"Successfully retrieved HTML content, length: {len(html_content)} characters")
        
        # 3. If product_names not provided, load specified field from item_info.json
        all_items = product_names if product_names else []
        
        if not all_items:
            item_info_path = THIS_DIR / 'item_info.json'
            if item_info_path.exists():
                with open(item_info_path, 'r', encoding='utf-8') as f:
                    try:
                        item_data = json.load(f)
                        # Modified here: use passed data_field parameter
                        all_items = [str(item.get(data_field, '')) for item in item_data if data_field in item and item.get(data_field)]
                        await ctx.info(f"Loaded {len(all_items)} entries of field '{data_field}' from item_info.json")
                    except json.JSONDecodeError:
                        await ctx.warning("Failed to parse item_info.json")
            else:
                await ctx.error(f"No data provided and item_info.json file not found")
                return False
        
        # Ensure enough items
        if len(all_items) < 2:
            await ctx.warning(f"Only {len(all_items)} valid {data_field} entries, at least 2 required")
            if not all_items:
                return False
        
        await ctx.info(f"Processing {len(all_items)} {data_field} entries: {all_items[:5]}{'...' if len(all_items) > 5 else ''}")
        
        # ================== New: Try exact match first ==================
        await ctx.info("\nStep 1: Try exact matching...")
        exact_match_tags = []
        
        # Try exact matching for each item
        for item in all_items:
            # 1. First try string exact match (strictest, requires single text node)
            exact_tag = soup.find(lambda t: t.string and t.string.strip().lower() == item.lower())
            if exact_tag:
                exact_match_tags.append(exact_tag)
                await ctx.info(f"  ✓ String exact match success: '{item}' -> <{exact_tag.name}>")
                continue
                
            # 2. Then try get_text() exact match (handles tags with child elements)
            exact_tag = soup.find(lambda t: t.get_text().strip().lower() == item.lower())
            if exact_tag:
                exact_match_tags.append(exact_tag)
                await ctx.info(f"  ✓ get_text exact match success: '{item}' -> <{exact_tag.name}>")
                continue
            
            # 3. Finally try partial match (loosest)
            partial_tag = soup.find(lambda t: t.get_text() and item.lower() in t.get_text().strip().lower())
            if partial_tag:
                exact_match_tags.append(partial_tag)
                await ctx.info(f"  ✓ Partial match success: '{item}' found in <{partial_tag.name}>")
        
        # If enough exact matches found, use them directly
        if len(exact_match_tags) >= 2:
            await ctx.info(f"\nExact match successfully found {len(exact_match_tags)} tags, skipping tokenization step")
            
            # Randomly select up to 2 tags for lowest common parent finding
            selected_tags = exact_match_tags
            if len(exact_match_tags) > 2:
                selected_tags = random.sample(exact_match_tags, 2)
                await ctx.info(f"Randomly selected 2 from {len(exact_match_tags)} exact match tags for common parent search")
            
            # Debug output of input tag info
            await ctx.info(f"\n===== Start finding lowest common parent of {len(selected_tags)} tags =====")
            for i, tag in enumerate(selected_tags, 1):
                await ctx.info(f"Input Tag {i}: <{tag.name}> - Text: {tag.get_text().strip()[:50]}")
                await ctx.info(f"  DOM Path: {get_dom_path(tag)}")
                
            # Find lowest common parent
            common_parent = _lowest_common_parent(selected_tags)
            
            if not common_parent:
                await ctx.warning("No common parent containing all tags found, try tokenization method")
            elif common_parent.name in ['head', 'body', 'html']:
                await ctx.warning(f"Exact match common parent is <{common_parent.name}>, too large, try tokenization method")
            else:
                await ctx.info(f"Exact match successfully found suitable common parent: <{common_parent.name}>")
                goto_process_common_parent = True
        else:
            await ctx.info(f"Exact match found only {len(exact_match_tags)} tags, not enough, continue with tokenization matching")
            goto_process_common_parent = False
        
        # ================== If exact match fails, try tokenization matching ==================
        if not goto_process_common_parent:
            await ctx.info("\nStep 2: Try tokenization matching...")
            # 4. Tokenize all items
            tokenized_items = [_tokenize(item) for item in all_items]
            
            # Find minimum token length to align positions
            min_token_length = min(len(tokens) for tokens in tokenized_items)
            await ctx.info(f"Minimum token length: {min_token_length}")
            
            # 5. Try matching by position, looking for similar DOM paths
            best_position = None
            best_position_tags = []
            best_similarity_score = 0.0
            
            for pos in range(min_token_length):
                # Get all tokens at current position
                current_tokens = [item_tokens[pos] for item_tokens in tokenized_items]
                await ctx.info(f"\nTrying tokens at position {pos+1}: {', '.join(current_tokens[:5])}{'...' if len(current_tokens) > 5 else ''}")
                
                # Find matching tags
                position_tags = []
                for i, token in enumerate(current_tokens):
                    tag = soup.find(lambda t: t.string and token.lower() in t.string.lower())
                    if tag:
                        position_tags.append(tag)
                
                await ctx.info(f"  Position {pos+1} found {len(position_tags)}/{len(current_tokens)} matching tags")
                
                # Only compare DOM paths if enough tags found
                if len(position_tags) >= 2:
                    dom_paths = [get_dom_path(tag) for tag in position_tags]
                    
                    # Compute average similarity
                    path_similarities = []
                    for i in range(len(dom_paths)):
                        for j in range(i+1, len(dom_paths)):
                            similarity = _similar_ratio(dom_paths[i], dom_paths[j])
                            path_similarities.append(similarity)
                    
                    avg_similarity = sum(path_similarities) / len(path_similarities) if path_similarities else 0
                    await ctx.info(f"  Average DOM path similarity: {avg_similarity:.4f}")
                    
                    if avg_similarity > best_similarity_score:
                        best_similarity_score = avg_similarity
                        best_position = pos
                        best_position_tags = position_tags
                        await ctx.info(f"  ✓ Updated best position to {pos+1}, similarity {avg_similarity:.4f}")
            
            # 6. If best position found, use tags at that position
            if best_position is not None and best_similarity_score > 0.6:
                await ctx.info(f"\nUsing tags at best position {best_position+1}, DOM path similarity: {best_similarity_score:.4f}")
                for i, tag in enumerate(best_position_tags[:5]):
                    await ctx.info(f"  Tag {i+1}: <{tag.name}> - {tag.get_text().strip()[:50]}")
                    await ctx.info(f"  DOM Path: {get_dom_path(tag)}")
                
                selected_tags = best_position_tags
                if len(best_position_tags) > 2:
                    selected_tags = random.sample(best_position_tags, 2)
                    await ctx.info(f"\nRandomly selected 2 from {len(best_position_tags)} tags for common parent search")
                
                await ctx.info(f"\n===== Start finding lowest common parent of {len(selected_tags)} tags =====")
                for i, tag in enumerate(selected_tags, 1):
                    await ctx.info(f"Input Tag {i}: <{tag.name}> - Text: {tag.get_text().strip()[:50]}")
                    await ctx.info(f"  DOM Path: {get_dom_path(tag)}")
                    
                common_parent = _lowest_common_parent(selected_tags)
                
                if not common_parent:
                    await ctx.warning("No common parent containing all tags found")
                    return False
                
                await ctx.info(f"Found lowest common parent: <{common_parent.name}>")
                
                if common_parent.name in ['head', 'body', 'html']:
                    await ctx.error(f"Common parent is <{common_parent.name}>, result too broad, locating failed")
                    return False
            else:
                await ctx.warning("No position with sufficient DOM path similarity found")
                await ctx.info("Fallback to traditional method, use all tokens for matching")
                
                sample_names = all_items[:3] if len(all_items) >= 3 else all_items
                paths = get_item_paths(soup, sample_names)
                await ctx.info(f"Number of matches found: {len(paths)}")
                
                majority_tags = filter_paths(paths)
                if not majority_tags:
                    await ctx.warning("No valid tags matched, skipping further processing.")
                    return False
                
                await ctx.info(f"Selected {len(majority_tags)} best matching tags")
                
                selected_tags = majority_tags
                if len(majority_tags) > 2:
                    selected_tags = random.sample(majority_tags, 2)
                    await ctx.info(f"Randomly selected 2 from {len(majority_tags)} tags for common parent search")
                
                common_parent = find_parent_with_multiple_descriptions(selected_tags)
                if not common_parent:
                    await ctx.warning("No common parent containing all descriptions found.")
                    return False

                if common_parent.name in ['head', 'body', 'html']:
                    await ctx.error(f"Common parent is <{common_parent.name}>, result too broad, locating failed")
                    return False
        
        # ================== Handle found common parent ==================
        await ctx.info(f"\nFinal parent element: <{common_parent.name}> - Preview: {common_parent.get_text().strip()[:100]}...")
        
        # 7. Iterate over children of common parent, extract and save to JSON
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
        
        await ctx.info(f"Common parent element has {child_count} child nodes")
        await ctx.info(f"Extracted {len(beautiful_soup)} valid child nodes")
        
        if beautiful_soup:
            out_path = THIS_DIR / "BeautifulSoup_Content.json"
            with open(out_path, "w", encoding="utf-8") as jf:
                json.dump(beautiful_soup, jf, ensure_ascii=False, indent=4)
            await ctx.info(f"Written to JSON: {out_path}")
            return True
        else:
            await ctx.warning("No valid child node content under the common parent element.")
            await ctx.warning("Possible reasons:")
            await ctx.warning("1. Common parent element is empty or only contains text nodes")
            await ctx.warning("2. Child nodes cannot be prettified (likely text nodes)")
            return False
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        await ctx.error(f"Error occurred during tag location: {str(e)}")
        await ctx.error(f"Error details: {error_trace}")
        return False


async def process_name_tag_location(ctx, file_path: str | None = None) -> bool:
    """
    Process locating product name tags
    """
    await ctx.info("Start processing product name tag location...")
    
    # Load product names from item_info.json
    try:
        product_names = await load_item_info(ctx, key='item')
        
        if not product_names:
            await ctx.warning("No 'item' field found, try using 'price' field")
            product_names = await load_item_info(ctx, key='price')
            
            if not product_names:
                await ctx.error("Neither 'item' nor 'price' field found")
                return False
        
        field_type = 'item' if 'item' in str(product_names[0]) else 'price'
        result = await process_tag_location(ctx, product_names, file_path, data_field=field_type)
        if result:
            await ctx.info("Product name tag location completed")
        else:
            await ctx.warning("Product name tag location failed")
        return result
                
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        await ctx.error(f"Error occurred during product name tag location: {str(e)}")
        await ctx.error(f"Error details: {error_trace}")
        return False


async def process_price_tag_location(ctx, file_path: str | None = None) -> bool:
    """
    Process locating product price tags
    """
    await ctx.info("Start processing product price tag location...")
    
    # Load price info from item_info.json
    try:
        price_info = await load_item_info(ctx, key='price')
        
        if not price_info:
            await ctx.error("No 'price' field found")
            return False
        
        result = await process_tag_location(ctx, price_info, file_path, data_field='price')
        if result:
            await ctx.info("Product price tag location completed")
        else:
            await ctx.warning("Product price tag location failed")
        return result
                
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        await ctx.error(f"Error occurred during product price tag location: {str(e)}")
        await ctx.error(f"Error details: {error_trace}")
        return False


# ────────────────────────────────────────────────────────────────────────────────
# CLI Entry
# ────────────────────────────────────────────────────────────────────────────────
class CliContext:
    """Context object for CLI tool, simulating MCP Context interface"""
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
