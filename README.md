# ODLP MCP

This MCP service uses a large language model (LLM) to extract price information from e-commerce websites.

## Environment Setup

The project manages Python dependencies with [uv](https://github.com/astral-sh/uv). To set up the environment:

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# create and activate virtual environment
uv venv
source .venv/bin/activate

# install project dependencies
uv sync
```

## API Key Configuration

`config.py` reads API keys and model names from environment variables. Create a `.env` file in the project root and add entries such as:

```
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_REASONING_MODEL=o4-mini
```

Use a general model (e.g., `OPENAI_MODEL`) to extract key elements and a reasoning model (e.g., `OPENAI_REASONING_MODEL`) to generate CSS selectors.

## Model and Service Customization

OpenAI is used by default. To switch to another provider (e.g., DeepSeek), update the API calls in `config.py`, `extractor/ocr.py`, and `extractor/css_selector_generator.py`, and supply the corresponding API key in `.env`. The OCR module can change services via the `service_type` parameter of `process_ocr_price`.

## Run the Service

After configuration, start the MCP server with:

```bash
uv run python server.py
```

## Connect to MCP Clients

To connect this server to an MCP-compatible client, follow the guidelines in the [VS Code Copilot MCP documentation](https://code.visualstudio.com/docs/copilot/customization/mcp-servers) and create an `MCP.json` file containing:

```json
{
  "servers": {
    "ODLP_MCP": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--project", "${workspaceFolder}",
        "--with", "mcp",
        "--with", "python-dotenv",
        "--with", "openai",
        "--with", "drissionpage",
        "--with", "beautifulsoup4",
        "--with", "playwright",
        "--with", "pytesseract",
        "mcp",
        "run",
        "/absolute/path/to/server.py"
      ]
    }
  }
}
```

Replace `/absolute/path/to/server.py` with the path to `server.py` on your machine.

