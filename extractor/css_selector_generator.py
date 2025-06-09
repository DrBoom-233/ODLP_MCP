#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSS选择器生成器
---------------
此模块负责：
1. 读取BeautifulSoup_Content.json中的HTML代码
2. 把用户输入的自然语言提取需求转换为结构化字段
3. 随机抽取HTML代码块发送给LLM
4. 让LLM生成CSS选择器
5. 使用Playwright执行提取任务
"""

import json
import random
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import glob

# 导入配置
import config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CssSelectorGenerator")

# 尝试导入openai库，如果未安装则提示安装
try:
    import openai
except ImportError:
    logger.error("请安装OpenAI库: pip install openai")
    raise

# 尝试导入playwright库，如果未安装则提示安装
try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    logger.error("请安装Playwright库: pip install playwright")
    raise

# 确保目录存在
def ensure_directory_exists(directory_path: Path):
    """确保指定的目录存在，如不存在则创建"""
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory_path}")

# 创建项目所需的目录
project_root = Path(__file__).parent.parent
SCHEMAS_DIR = project_root / "extraction_schemas"
OUTPUT_DIR = project_root / "price_info_output"

# 确保目录存在
ensure_directory_exists(SCHEMAS_DIR)
ensure_directory_exists(OUTPUT_DIR)

class CssSelectorGenerator:
    """CSS选择器生成器类，用于与LLM交互并生成CSS选择器"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化CSS选择器生成器
        
        参数:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: LLM模型名称，如果为None则从环境变量获取
        """
        # 从配置或参数获取API密钥和模型
        self.api_key = api_key or config.API_KEY
        self.model = model or config.REASONING_MODEL
        self.url = config.URL
        
        if not self.api_key:
            raise ValueError("请提供OpenAI API密钥或设置OPENAI_API_KEY环境变量")
            
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.url)
        
        # 指定项目路径
        self.base_path = Path(__file__).parent
        self.html_data_path = self.base_path / "BeautifulSoup_Content.json"
        
        # 检查文件是否存在
        if not self.html_data_path.exists():
            raise FileNotFoundError(f"HTML数据文件不存在: {self.html_data_path}")
            
        logger.info(f"初始化CSS选择器生成器，使用模型: {self.model}")

    def load_html_data(self) -> List[Dict[str, Any]]:
        """
        加载BeautifulSoup_Content.json中的HTML数据
        
        返回:
            HTML数据列表
        """
        try:
            with open(self.html_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载HTML数据，共{len(data)}条记录")
            return data
        except Exception as e:
            logger.error(f"加载HTML数据失败: {e}")
            raise

    def sample_html_blocks(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        从HTML数据中随机抽取指定数量的HTML代码块
        
        参数:
            count: 要抽取的HTML代码块数量
            
        返回:
            抽取的HTML代码块列表
        """
        data = self.load_html_data()
        # 只选择有HTML内容的项目
        valid_data = [item for item in data if item.get("Content")]
        
        if len(valid_data) < count:
            logger.warning(f"数据不足，请求{count}个样本但只有{len(valid_data)}个可用")
            return valid_data
            
        # 随机抽样
        samples = random.sample(valid_data, count)
        logger.info(f"成功抽取{len(samples)}个HTML样本")
        return samples

    async def natural_language_to_fields(self, natural_language_request: str) -> List[Dict[str, str]]:
        """
        将自然语言提取需求转换为结构化的提取字段列表
        
        参数:
            natural_language_request: 用户的自然语言提取需求
            
        返回:
            提取字段列表，例如 [{"name": "price"}, {"name": "product_name"}]
        """
        logger.info("将自然语言需求转换为结构化字段...")
        
        prompt = f"""
请将以下自然语言提取需求转换为结构化的字段列表。

用户需求: {natural_language_request}

请输出一个JSON数组，每个元素包含一个"name"字段，表示要提取的数据字段名称。
例如，如果用户想要提取商品价格和名称，输出应该是:
[
  {{"name": "product_name"}},
  {{"name": "price"}}
]

只需要输出JSON数组，不需要其他解释。确保每个字段名称简洁、准确，并与用户需求相关。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个数据提取专家，能够将自然语言需求转换为结构化的提取字段。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # 低温度以获得更确定性的回答
            )
            
            result_text = response.choices[0].message.content
            
            # 尝试提取JSON数组
            try:
                # 查找JSON部分
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|(\[[\s\S]*\])', result_text)
                
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    extraction_fields = json.loads(json_str)
                else:
                    # 尝试直接解析整个文本
                    extraction_fields = json.loads(result_text)
                
                # 验证结果格式
                if not isinstance(extraction_fields, list):
                    raise ValueError("LLM输出的不是有效的字段列表")
                    
                for field in extraction_fields:
                    if not isinstance(field, dict) or "name" not in field:
                        raise ValueError("字段列表中的元素格式不正确")
                
                logger.info(f"成功将自然语言转换为{len(extraction_fields)}个字段: {extraction_fields}")
                return extraction_fields
                
            except json.JSONDecodeError as e:
                logger.error(f"解析LLM响应中的JSON失败: {e}")
                logger.debug(f"LLM响应: {result_text}")
                raise ValueError(f"无法从LLM响应中提取有效的JSON: {e}")
                
        except Exception as e:
            logger.error(f"调用LLM API失败: {e}")
            raise

    async def generate_css_selectors(self, extraction_fields: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        根据用户需求和HTML样本生成CSS选择器
        
        参数:
            extraction_fields: 提取字段列表，例如[{"name": "price"}, {"name": "title"}]
            
        返回:
            含有CSS选择器的字典
        """
        # 获取HTML样本
        html_samples = self.sample_html_blocks(3)
        
        # 构建提示
        prompt = f"""
分析以下HTML代码，为指定的字段生成精确的CSS选择器。

提取字段:
{json.dumps(extraction_fields, indent=2, ensure_ascii=False)}

HTML样本 (随机选择的3个):
"""

        # 添加HTML样本到提示中
        for i, sample in enumerate(html_samples):
            prompt += f"\n\nHTML样本 {i+1}:\n```html\n{sample['Content']}\n```"
            
        prompt += """
根据上述HTML和提取字段，请生成一个包含以下信息的JSON:
1. website_type: 网站类型描述
2. description: 提取任务简要描述
3. container_selector: 用于选择包含所有字段的父容器元素的CSS选择器（例如，如果要提取商品列表中的每个商品，这个选择器应该能选中单个商品项容器）
4. expected_fields: 包含每个字段名称和对应的CSS选择器

输出格式应为:
{
  "website_type": "类型描述",
  "description": "提取任务描述",
  "container_selector": ".product-item, .item, .product, li.product, div.product, [class*='product-'], [class*='item-']",
  "expected_fields": [
    {
      "name": "字段1",
      "selector": "CSS选择器1"
    },
    ...
  ]
}

container_selector字段非常重要，它应该选择包含所有需要提取字段的容器元素。这可以确保提取的数据之间保持正确的关联关系（如商品名称和价格匹配）。
为每个字段提供尽可能准确的CSS选择器，确保可以通过Playwright提取到相应内容。
"""

        # 调用OpenAI API
        try:
            logger.info("正在请求LLM生成CSS选择器...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的网页数据提取专家，精通HTML分析和CSS选择器编写。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2  # 低温度以获得更确定性的回答
            )
            
            # 提取回答文本
            result_text = response.choices[0].message.content
            
            # 尝试从回答中提取JSON
            try:
                # 查找JSON部分
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', result_text)
                
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    result = json.loads(json_str)
                else:
                    # 尝试直接解析整个文本
                    result = json.loads(result_text)
                
                # 确保配置包含container_selector字段
                if "container_selector" not in result:
                    logger.warning("生成的选择器配置中缺少container_selector，添加默认值")
                    result["container_selector"] = ".product-item, .item, .product, li.product, div.product, [class*='product-'], [class*='item-']"
                
                logger.info("成功生成CSS选择器")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"解析LLM响应中的JSON失败: {e}")
                logger.debug(f"LLM响应: {result_text}")
                raise ValueError(f"无法从LLM响应中提取有效的JSON: {e}")
                
        except Exception as e:
            logger.error(f"调用LLM API失败: {e}")
            raise

    async def extract_with_playwright(self, selectors_config: Dict[str, Any], html_content: str) -> Dict[str, Any]:
        """
        使用Playwright和生成的CSS选择器从HTML中提取数据
        
        参数:
            selectors_config: 包含CSS选择器的配置
            html_content: 要提取数据的HTML内容
            
        返回:
            提取的数据
        """
        # 创建临时HTML文件
        temp_html_path = self.base_path / "temp_extraction.html"
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        results = {}
        
        # 使用Playwright提取数据
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                # 导航到临时HTML文件
                await page.goto(f"file://{temp_html_path.resolve()}")
                
                # 对每个字段使用CSS选择器提取内容
                for field in selectors_config.get("expected_fields", []):
                    field_name = field.get("name")
                    selector = field.get("selector")
                    
                    if not field_name or not selector:
                        continue
                        
                    try:
                        # 等待选择器
                        await page.wait_for_selector(selector, timeout=5000)
                        
                        # 提取内容
                        elements = await page.query_selector_all(selector)
                        if elements:
                            # 提取文本内容
                            texts = []
                            for element in elements:
                                text = await element.text_content()
                                if text:
                                    texts.append(text.strip())
                            
                            # 保存结果
                            results[field_name] = texts if len(texts) > 1 else texts[0] if texts else None
                        else:
                            results[field_name] = None
                            logger.warning(f"未找到匹配选择器的元素: {selector}")
                            
                    except Exception as e:
                        logger.error(f"提取字段 '{field_name}' 失败: {e}")
                        results[field_name] = None
                
            except Exception as e:
                logger.error(f"Playwright提取失败: {e}")
                raise
            finally:
                await browser.close()
                
                # 清理临时文件
                if temp_html_path.exists():
                    temp_html_path.unlink()
        
        return results

async def process_natural_language_request(natural_language_request: str, html_content: Optional[str] = None) -> Dict[str, Any]:
    """
    处理自然语言提取请求
    
    参数:
        natural_language_request: 自然语言形式的提取需求
        html_content: 可选的HTML内容，如果不提供则仅生成CSS选择器
        
    返回:
        处理结果
    """
    try:
        # 创建选择器生成器
        generator = CssSelectorGenerator()
        
        # 第一步：将自然语言需求转换为提取字段
        extraction_fields = await generator.natural_language_to_fields(natural_language_request)
        
        # 第二步：生成CSS选择器
        selectors_config = await generator.generate_css_selectors(extraction_fields)
        
        # 从mhtml_output目录中获取最新的mhtml文件名
        mhtml_dir = project_root / "mhtml_output"
        mhtml_files = list(glob.glob(str(mhtml_dir / "*.mhtml")))
        
        if mhtml_files:
            # 按修改时间排序，获取最新的文件
            latest_mhtml = max(mhtml_files, key=os.path.getmtime)
            # 提取文件名（不含扩展名）
            base_filename = os.path.basename(latest_mhtml).replace(".mhtml", "")
            logger.info(f"使用mhtml文件命名格式: {base_filename}")
        else:
            # 如果没有mhtml文件，使用之前的命名格式
            # 生成日期格式
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            
            # 获取网站类型，清理格式
            website_type = selectors_config.get("website_type", "unknown")
            # 将空格替换为下划线，移除特殊字符
            website_type = website_type.replace(" ", "_").replace("-", "_")
            website_type = ''.join(c for c in website_type if c.isalnum() or c == '_')
            
            # 获取描述作为类别，清理格式
            category = selectors_config.get("description", "general")
            # 将空格替换为短横线，移除特殊字符
            category = category.replace(" ", "-").replace("_", "-")
            category = ''.join(c for c in category if c.isalnum() or c == '-')
            
            base_filename = f"{website_type}_{category}_{current_date}"
            logger.info(f"未找到mhtml文件，使用生成的命名格式: {base_filename}")
        
        # 保存CSS选择器配置到extraction_schemas目录
        schema_filename = f"{base_filename}.json"
        schema_path = SCHEMAS_DIR / schema_filename
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(selectors_config, f, indent=2, ensure_ascii=False)
        logger.info(f"CSS选择器配置已保存到: {schema_path}")
        
        # 如果提供了HTML内容，则使用Playwright提取数据
        if html_content:
            extraction_results = await generator.extract_with_playwright(selectors_config, html_content)
            
            # 保存提取结果到price_info_output目录
            results_filename = f"{base_filename}_results.json"
            results_path = OUTPUT_DIR / results_filename
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_results, f, indent=2, ensure_ascii=False)
            logger.info(f"提取结果已保存到: {results_path}")
            
            return {
                "selectors_config": selectors_config,
                "extraction_results": extraction_results,
                "schema_path": str(schema_path),
                "results_path": str(results_path)
            }
        
        return {
            "selectors_config": selectors_config,
            "schema_path": str(schema_path)
        }
        
    except Exception as e:
        logger.error(f"处理提取请求失败: {e}")
        return {"error": str(e)}

# 保持向后兼容的接口
async def process_extraction_request(extraction_fields: List[Dict[str, str]], html_content: Optional[str] = None) -> Dict[str, Any]:
    """
    处理结构化提取请求的主函数（向后兼容）
    
    参数:
        extraction_fields: 需要提取的字段列表 
        html_content: 可选的HTML内容，如果不提供则使用生成的CSS选择器
        
    返回:
        处理结果
    """
    try:
        # 创建选择器生成器
        generator = CssSelectorGenerator()
        
        # 生成CSS选择器
        selectors_config = await generator.generate_css_selectors(extraction_fields)
        
        # 从mhtml_output目录中获取最新的mhtml文件名
        mhtml_dir = project_root / "mhtml_output"
        mhtml_files = list(glob.glob(str(mhtml_dir / "*.mhtml")))
        
        if mhtml_files:
            # 按修改时间排序，获取最新的文件
            latest_mhtml = max(mhtml_files, key=os.path.getmtime)
            # 提取文件名（不含扩展名）
            base_filename = os.path.basename(latest_mhtml).replace(".mhtml", "")
            logger.info(f"使用mhtml文件命名格式: {base_filename}")
        else:
            # 如果没有mhtml文件，使用之前的命名格式
            # 生成日期格式
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            
            # 获取网站类型，清理格式
            website_type = selectors_config.get("website_type", "unknown")
            # 将空格替换为下划线，移除特殊字符
            website_type = website_type.replace(" ", "_").replace("-", "_")
            website_type = ''.join(c for c in website_type if c.isalnum() or c == '_')
            
            # 获取描述作为类别，清理格式
            category = selectors_config.get("description", "general")
            # 将空格替换为短横线，移除特殊字符
            category = category.replace(" ", "-").replace("_", "-")
            category = ''.join(c for c in category if c.isalnum() or c == '-')
            
            base_filename = f"{website_type}_{category}_{current_date}"
            logger.info(f"未找到mhtml文件，使用生成的命名格式: {base_filename}")
        
        # 保存CSS选择器配置到extraction_schemas目录
        schema_filename = f"{base_filename}.json"
        schema_path = SCHEMAS_DIR / schema_filename
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(selectors_config, f, indent=2, ensure_ascii=False)
        logger.info(f"CSS选择器配置已保存到: {schema_path}")
        
        # 如果提供了HTML内容，则使用Playwright提取数据
        if html_content:
            extraction_results = await generator.extract_with_playwright(selectors_config, html_content)
            
            # 保存提取结果到price_info_output目录
            results_filename = f"{base_filename}_results.json"
            results_path = OUTPUT_DIR / results_filename
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_results, f, indent=2, ensure_ascii=False)
            logger.info(f"提取结果已保存到: {results_path}")
            
            return {
                "selectors_config": selectors_config,
                "extraction_results": extraction_results,
                "schema_path": str(schema_path),
                "results_path": str(results_path)
            }
        
        return {
            "selectors_config": selectors_config,
            "schema_path": str(schema_path)
        }
        
    except Exception as e:
        logger.error(f"处理提取请求失败: {e}")
        return {"error": str(e)}

async def main():
    """主函数，用于测试"""
    # 测试自然语言请求
    natural_request = "我想要提取所有商品的名称、价格和单位"
    
    print("测试自然语言请求处理：")
    result = await process_natural_language_request(natural_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 也可以测试结构化字段请求（向后兼容）
    test_fields = [
        {"name": "product_name"},
        {"name": "price"},
        {"name": "unit"}
    ]
    
    print("\n测试结构化字段请求处理（向后兼容）：")
    result2 = await process_extraction_request(test_fields)
    print(json.dumps(result2, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main()) 