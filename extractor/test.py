# test_sync_main.py (放在项目根目录下)
import subprocess
import sys
from pathlib import Path

# MAIN_SCRIPTS 从 pipeline.py 里拿到脚本列表
from extractor.pipeline import MAIN_SCRIPTS

def main():
    project_root = Path(__file__).parent.parent.resolve()
    python_exe = sys.executable

    # 确保 cwd 在项目根，以便 Python 把 project_root 当作顶级包目录
    for exe, rel_path, need_file in MAIN_SCRIPTS:
        module_name = "extractor." + Path(rel_path).stem  # e.g. "extractor.image_transform_2"

        # 构造基础命令
        cmd = [python_exe, "-m", module_name]

        # # 如果脚本需要 mhtml 文件，就给它一个 dummy（或者改成你真实的文件路径）
        if need_file:
            dummy = project_root / "mhtml_output" / "dummy.mhtml"
            dummy.parent.mkdir(exist_ok=True)
            if not dummy.exists():
                dummy.write_text("", encoding="utf-8")
            cmd.append(str(dummy))

        print(f"\n▶ Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=str(project_root), check=True)
            print(f"✅ {module_name} exited with 0")
        except subprocess.CalledProcessError as e:
            print(f"❌ {module_name} failed (exit {e.returncode})")
            # 如果你想看到脚本的 stderr，可以去掉下面这行的注释
            # print(e.stderr)
            sys.exit(e.returncode)

if __name__ == "__main__":
    main()
