from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

def main() -> None:
    if len(sys.argv) != 2:
        print("用法: python screenshot.py <本地 HTML/MHTML 路径>")
        sys.exit(1)

    file_path = Path(sys.argv[1]).expanduser().resolve()
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        sys.exit(1)

    # 转成 file:// URI，Playwright 才能识别
    target_uri = file_path.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--allow-file-access-from-files",  # 解除 file:// 跨域限制
                "--disable-web-security",
                "--start-maximized",
            ],

        )

        context = browser.new_context(
            viewport={"width": 2400, "height": 2400, "device_scale_factor": 1}
        )
        page = context.new_page()

        page.goto(target_uri, wait_until="domcontentloaded", timeout=10_000)

        # 只做一次滚动到页底即可
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)  # 保险等待 1.5 s

        project_root = Path(__file__).resolve().parent.parent
        out_dir = project_root / "public"
        out_dir.mkdir(exist_ok=True)
        page.screenshot(path=str(out_dir / "screenshot.png"), full_page=True)

        print("Done ✅")
        browser.close()

if __name__ == "__main__":
    main()
