# 🚀 ODLP MCP

> A lightweight MCP service that uses a large language model (LLM) to extract price and product information from e‑commerce pages.

---

✨ Highlights
- Adapts to many e-commerce layouts and page structures
- Extracts fields like product name, price, size, color, country/region, etc.
- No hardcoded rules, just describe your needs in natural language.
---

How it works
- Describe your extraction needs in natural language (e.g., “Extract product title, current price, available sizes, shipping country, and return JSON”).
- The LLM plans steps, generates CSS/XPath selectors, and performs extraction automatically.

---

## 🔧 Before You Start
- Save a page as MHTML: in your browser use "Save as" → "Webpage, Single File (*.mhtml)". To ensure extraction success, please go to **category specific grid-style product listing pages**, where the DOM Tree traversing algorithm can work. Example url like: `https://www.metro.ca/en/online-grocery/aisles/fruits-vegetables`. Not like `https://www.amazon.ca/`.
- Obtain an LLM API key (e.g., OpenAI). Recommended: one general model (e.g., `gpt-4o-mini`) and one reasoning model (e.g., `o4-mini`).

---

## 🛠 Environment Setup

The project manages Python dependencies with [uv](https://github.com/astral-sh/uv).

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# create and activate virtual environment
uv venv
source .venv/bin/activate

# install project dependencies
uv sync
```

---

## 🔑 API Key Configuration

`config.py` reads API keys and model names from environment variables. Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_REASONING_MODEL=o4-mini
```

Use `OPENAI_MODEL` for general extraction and `OPENAI_REASONING_MODEL` for selector/reasoning tasks.

---

## 🔁 Model & Service Customization

OpenAI is the default. To switch providers (e.g., DeepSeek), update API calls in:
- `config.py`
- `extractor/ocr.py`
- `extractor/css_selector_generator.py`

OCR service can be changed via the `service_type` parameter of `process_ocr_price`.

---

## 🔌 Connect to MCP Clients

Create an `MCP.json` to register this server with an MCP-compatible client:

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

Replace `/absolute/path/to/server.py` with the absolute path to `server.py` on your machine.

---

⭐ From my side I use GitHub Copilot in VS Code as the client. Tutorial website: https://code.visualstudio.com/docs/copilot/customization/mcp-servers; you may use other MCP-compatible clients (Claude Code, Gemini CLI, etc.).
