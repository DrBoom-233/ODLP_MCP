#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取执行器
---------
该模块负责执行数据提取操作，基于提供的选择器配置从mhtml文件中提取数据
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExtractionExecutor")

# 导入Playwright
try:
    from playwright.async_api import Page, Browser
except ImportError:
    logger.error("请安装Playwright库: pip install playwright")
    raise

class ExtractionExecutor:
    """
    提取执行器类：负责使用配置的选择器从MHTML文件中提取数据
    """

    def __init__(self, browser: Browser, info_callback: Optional[Callable] = None, error_callback: Optional[Callable] = None):
        """
        初始化提取执行器
        
        参数:
            browser: Playwright浏览器实例
            info_callback: 信息回调函数，用于发送信息通知
            error_callback: 错误回调函数，用于发送错误通知
        """
        self.browser = browser
        self.info_callback = info_callback or (lambda msg: logger.info(msg))
        self.error_callback = error_callback or (lambda msg: logger.error(msg))

    async def load_selector_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载选择器配置文件
        
        参数:
            config_path: 配置文件路径
            
        返回:
            解析后的配置字典
        """
        path = Path(config_path)
        if not path.exists() or not path.is_file():
            error_msg = f"配置文件不存在: {config_path}"
            self.error_callback(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.info_callback("✅ 成功加载配置文件")
            
            # 显示配置文件信息
            self.info_callback(f"📋 网站类型: {config.get('website_type', '未指定')}")
            self.info_callback(f"📝 描述: {config.get('description', '未提供')}")
            
            # 输出提取字段信息
            fields = config.get("expected_fields", [])
            if fields:
                field_names = [field.get("name", "") for field in fields]
                self.info_callback(f"🔍 提取字段: {', '.join(field_names)}")
                
            # 检查容器选择器
            if "container_selector" not in config:
                self.error_callback("⚠️ 配置中未指定容器选择器，将尝试使用默认容器选择器")
                config["container_selector"] = ".product-item, .item, .product, li.product, div.product, [class*='product-'], [class*='item-']"
                
                # 更新配置文件
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                self.info_callback(f"✅ 已添加默认容器选择器: {config['container_selector']}")
            
            return config
            
        except Exception as e:
            error_msg = f"读取配置文件失败: {str(e)}"
            self.error_callback(f"❌ {error_msg}")
            raise ValueError(error_msg)

    async def find_mhtml_files(self, directory: str = "mhtml_output") -> List[Path]:
        """
        查找MHTML文件
        
        参数:
            directory: 目录路径
            
        返回:
            MHTML文件路径列表
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            error_msg = f"目录不存在: {directory}"
            self.error_callback(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
            
        mhtml_files = list(dir_path.glob("*.mhtml"))
        if not mhtml_files:
            error_msg = f"{directory}目录中没有找到mhtml文件"
            self.error_callback(f"❌ {error_msg}")
            raise FileNotFoundError(error_msg)
            
        self.info_callback(f"🔍 找到 {len(mhtml_files)} 个MHTML文件")
        return mhtml_files

    async def extract_from_file(self, mhtml_file: Path, selectors_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        从单个MHTML文件中提取数据
        
        参数:
            mhtml_file: MHTML文件路径
            selectors_config: 选择器配置
            
        返回:
            提取结果字典
        """
        self.info_callback(f"📄 处理MHTML文件: {mhtml_file.name}")
        
        try:
            # 创建新页面
            page = await self.browser.new_page()
            
            try:
                # 导航到mhtml文件
                file_url = f"file://{mhtml_file.absolute()}"
                self.info_callback(f"🌐 加载文件: {file_url}")
                await page.goto(file_url)
                await page.wait_for_load_state("networkidle")
                
                # 获取配置信息
                container_selector = selectors_config.get("container_selector", "")
                fields = selectors_config.get("expected_fields", [])
                
                # 使用容器选择器提取所有项目容器
                self.info_callback(f"🔍 查找项目容器，使用选择器: {container_selector}")
                item_elements = await page.query_selector_all(container_selector)
                
                if not item_elements:
                    self.error_callback(f"⚠️ 未找到任何项目容器，将直接从页面提取")
                    
                    # 如果找不到容器，则尝试直接提取每个字段作为独立项
                    product_items = []
                    
                    # 对每个字段类型分别提取
                    field_values = {}
                    for field in fields:
                        field_name = field.get("name", "")
                        selector = field.get("selector", "")
                        
                        if not field_name or not selector:
                            continue
                            
                        try:
                            elements = await page.query_selector_all(selector)
                            if elements:
                                texts = []
                                for element in elements:
                                    text = await element.text_content()
                                    if text:
                                        texts.append(text.strip())
                                
                                field_values[field_name] = texts
                                self.info_callback(f"✅ 找到 {len(texts)} 个 {field_name}")
                            else:
                                field_values[field_name] = []
                        except Exception as e:
                            self.error_callback(f"❌ 提取字段 {field_name} 失败: {str(e)}")
                            field_values[field_name] = []
                    
                    # 将不同字段的结果配对成产品项
                    max_items = max([len(values) for values in field_values.values()]) if field_values else 0
                    
                    for idx in range(max_items):
                        item = {}
                        for field_name, values in field_values.items():
                            item[field_name] = values[idx] if idx < len(values) else ""
                        product_items.append(item)
                        
                    self.info_callback(f"📊 成功配对 {len(product_items)} 个产品项")
                else:
                    self.info_callback(f"✅ 找到 {len(item_elements)} 个项目容器")
                    
                    # 处理每个容器
                    product_items = []
                    for idx, element in enumerate(item_elements):
                        item_data = {}
                        
                        # 对每个字段在容器内提取内容
                        for field in fields:
                            field_name = field.get("name", "")
                            field_selector = field.get("selector", "")
                            
                            if not field_name or not field_selector:
                                continue
                            
                            try:
                                # 在容器内查找元素
                                sub_elem = await element.query_selector(field_selector)
                                if sub_elem:
                                    raw = await sub_elem.text_content()
                                    text = raw.strip() if raw and raw.strip() else None

                                    # 2) 如果文本是空，再试 aria-label 属性
                                    if not text:
                                        raw_attr = await sub_elem.get_attribute("aria-label")
                                        text = raw_attr.strip() if raw_attr and raw_attr.strip() else None
                                    
                                    # 3) 最终赋值（都没有时设为 ""）
                                    item_data[field_name] = text or ""
                                else:
                                    item_data[field_name] = ""                                   
                            except Exception as e:
                                self.error_callback(f"❌ 容器 #{idx+1} 中提取字段 '{field_name}' 失败: {str(e)}")
                                item_data[field_name] = ""
                        
                        # 添加到结果
                        product_items.append(item_data)
                
                # 显示部分提取结果
                if product_items:
                    self.info_callback(f"📊 成功提取 {len(product_items)} 个产品项")
                    if len(product_items) > 0:
                        sample = product_items[0]
                        sample_str = ", ".join([f"{k}: {v}" for k, v in sample.items()])
                        self.info_callback(f"📌 样例: {sample_str}")
                
                # 返回该文件的结果
                return {
                    "file_name": mhtml_file.name,
                    "items_count": len(product_items),
                    "items": product_items
                }
                
            except Exception as e:
                self.error_callback(f"❌ 处理MHTML文件失败: {str(e)}")
                return {
                    "file_name": mhtml_file.name,
                    "items_count": 0,
                    "items": [],
                    "error": str(e)
                }
            finally:
                await page.close()
                
        except Exception as e:
            self.error_callback(f"❌ 创建页面失败: {str(e)}")
            return {
                "file_name": mhtml_file.name,
                "items_count": 0,
                "items": [],
                "error": str(e)
            }

    async def execute_extraction(self, selectors_config_path: str, output_dir: str = "price_info_output") -> Dict[str, Any]:
        """
        执行提取操作的主函数
        
        参数:
            selectors_config_path: 选择器配置文件路径
            output_dir: 输出目录
            
        返回:
            包含提取结果的字典
        """
        try:
            # 加载选择器配置
            selectors_config = await self.load_selector_config(selectors_config_path)
            
            # 查找mhtml文件
            mhtml_files = await self.find_mhtml_files()
            
            # 准备提取操作
            self.info_callback("🔍 开始提取数据...")
            
            # 处理所有文件
            all_files_results = []
            for mhtml_file in mhtml_files:
                file_result = await self.extract_from_file(mhtml_file, selectors_config)
                all_files_results.append(file_result)
            
            # 使用MHTML文件名作为输出JSON名称
            if len(mhtml_files) == 1:
            # 如果只有一个MHTML文件，直接使用其名称
                mhtml_name = mhtml_files[0].stem  # 获取不带扩展名的文件名
                results_filename = f"{mhtml_name}.json"
            else:
            # 如果有多个MHTML文件，使用第一个文件名并添加指示
                mhtml_name = mhtml_files[0].stem
                results_filename = f"{mhtml_name}_and_{len(mhtml_files)-1}_more.json"
            
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 结果文件路径
            results_path = output_path / results_filename
            
            # 计算总项目数
            total_items = sum(file_result.get("items_count", 0) for file_result in all_files_results)
            
            # 构建最终结果对象
            final_results = {
                "files_processed": len(all_files_results),
                "total_items": total_items,
                "results": all_files_results
            }
            
            # 保存结果
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            self.info_callback(f"💾 提取结果已保存至: {results_path}")
            
            # 展示总结果
            self.info_callback(f"📊 已成功处理 {len(all_files_results)}/{len(mhtml_files)} 个MHTML文件，共提取 {total_items} 个数据项")
                
            return {
                "success": True,
                "files_processed": len(all_files_results),
                "total_items": total_items,
                "results_path": str(results_path)
            }
            
        except Exception as e:
            self.error_callback(f"❌ 数据提取过程出错: {str(e)}")
            return {"success": False, "error": str(e)}

# 导出简化的异步函数，供server.py调用
async def execute_extraction(
    browser: Browser, 
    selectors_config_path: str,
    info_callback: Optional[Callable] = None,
    error_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    执行数据提取的便捷函数
    
    参数:
        browser: Playwright浏览器实例
        selectors_config_path: 选择器配置文件路径
        info_callback: 信息回调函数
        error_callback: 错误回调函数
        
    返回:
        包含提取结果的字典
    """
    executor = ExtractionExecutor(browser, info_callback, error_callback)
    return await executor.execute_extraction(selectors_config_path) 