"""Generates the authoritative ASCII directory tree for README.md directly from the filesystem."""
from pathlib import Path
import sys

# Configure UTF-8 stdout on Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {".git", ".github", "__pycache__", ".pytest_cache", ".venv", "venv", "egg-info"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".png"}

def build_tree_text(dir_path: Path, prefix: str = "", max_depth: int = 2, current_depth: int = 0) -> str:
    if current_depth > max_depth:
        return ""
    
    lines = []
    items = sorted(
        [p for p in dir_path.iterdir() if p.name not in EXCLUDE_DIRS and p.suffix not in EXCLUDE_EXTS],
        key=lambda x: (not x.is_dir(), x.name.lower())
    )

    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "+-- "
        child_prefix = "    " if is_last else "|   "
        
        lines.append(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")
        if item.is_dir() and current_depth < max_depth:
            sub = build_tree_text(item, prefix + child_prefix, max_depth, current_depth + 1)
            if sub:
                lines.append(sub)
                
    return "\n".join(lines)

def main():
    tree = f"TrustRisk-WDS/\n" + build_tree_text(ROOT, max_depth=2)
    print(tree)
    return tree

if __name__ == "__main__":
    main()
