from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import config
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Any

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="uv",  # Executable
    args=[
        "run",
        "--with", "mcp",
        "--with", "python-dotenv",
        "--with", "pytesseract",
        "--with", "openai",
        "--with", "beautifulsoup4",
        "--with", "playwright",
        "--with", "drissionpage",
        "mcp", "run", "server.py"
    ],  # Optional command line arguments
    env=None,  # Optional environment variables
)


# Optional: create a sampling callback
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model=config.CHAT_MODEL,
        stopReason="endTurn",
    )


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # Initialize the connection
            await session.initialize()

            print("连接初始化成功")

            # 列出所有可用工具
            tools = await session.list_tools()
            print("可用工具列表:")
            for tool in tools:
                print(f" - {tool}")

            # 示例1: 下载URL
            urls_to_download = ["https://www.foodbasics.ca/dairy-eggs"]
            try:
                download_result = await session.call_tool(
                    "download_urls_tool",
                    arguments={"urls": urls_to_download}
                )
                print(f"下载结果: {download_result}")
            except Exception as e:
                print(f"下载URL工具调用失败: {e}")

            # 示例2: 生成截图
            try:
                screenshot_result = await session.call_tool("screenshot_tool")
                print(f"截图结果: {screenshot_result}")
            except Exception as e:
                print(f"截图工具调用失败: {e}")

            # 示例3: OCR名称识别
            try:
                ocr_name_result = await session.call_tool(
                    "ocr_name_tool",
                    arguments={"image_path": None}  # 可以指定图片路径，不指定则使用最新的
                )
                print(f"OCR名称识别结果: {ocr_name_result}")
            except Exception as e:
                print(f"OCR名称识别工具调用失败: {e}")

            # 示例4: OCR价格识别
            try:
                ocr_price_result = await session.call_tool(
                    "ocr_price_tool",
                    arguments={"image_path": None}  # 可以指定图片路径，不指定则使用最新的
                )
                print(f"OCR价格识别结果: {ocr_price_result}")
            except Exception as e:
                print(f"OCR价格识别工具调用失败: {e}")

            # 示例5: 名称标签定位
            try:
                name_locating_result = await session.call_tool("name_tag_locating_tool")
                print(f"名称标签定位结果: {name_locating_result}")
            except Exception as e:
                print(f"名称标签定位工具调用失败: {e}")

            # 示例6: 价格标签定位
            try:
                price_locating_result = await session.call_tool("price_tag_locating_tool")
                print(f"价格标签定位结果: {price_locating_result}")
            except Exception as e:
                print(f"价格标签定位工具调用失败: {e}")

            # 示例7: 最终总结
            try:
                summary_result = await session.call_tool("final_summary_tool")
                print(f"总结结果: {summary_result}")
            except Exception as e:
                print(f"最终总结工具调用失败: {e}")

            # 示例8: 数据提取配置生成
            try:
                extract_config_result = await session.call_tool(
                    "extract_data_tool",
                    arguments={
                        "page_url": "https://www.foodbasics.ca/dairy-eggs",
                        "sample_items": ["牛奶", "鸡蛋", "奶酪"],
                        "target_attributes": ["名称", "价格", "重量"]
                    }
                )
                print(f"数据提取配置生成结果: {extract_config_result}")
            except Exception as e:
                print(f"数据提取配置生成工具调用失败: {e}")

            # 示例9: 执行数据提取
            try:
                execute_extraction_result = await session.call_tool(
                    "execute_extraction_tool",
                    arguments={"selectors_config_path": ""}  # 空字符串表示使用最新的配置文件
                )
                print(f"执行数据提取结果: {execute_extraction_result}")
            except Exception as e:
                print(f"执行数据提取工具调用失败: {e}")


if __name__ == "__main__":
    asyncio.run(run())