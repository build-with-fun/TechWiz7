"""Count comment and docstring lines in the project's own source (not .venv).

    python tools/comment_stats.py [--json out.json]

Shows which files have the most comments.
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_DIRS = ["src", "audio_preprocessing", "feature_extraction", "python_models", "augmentation",
           "audio_dataset", "tools", "database", "tests", "data", "documentation"]
SKIP = {".venv", "__pycache__", ".git"}


def py_counts(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8")
    comments = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and not tok.string.startswith("#!"):
                comments += 1
    except (tokenize.TokenError, IndentationError):
        pass
    docs = 0
    try:
        tree = ast.parse(text)
        for node in [tree, *ast.walk(tree)]:
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docs += doc.count("\n") + 1
    except SyntaxError:
        pass
    return comments, docs, text.count("\n") + 1


def web_counts(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    lines = 0
    for pattern in (r"/\*.*?\*/", r"\{#.*?#\}", r"<!--.*?-->"):
        for m in re.finditer(pattern, text, flags=re.S):
            lines += m.group(0).count("\n") + 1
    if path.suffix == ".js":
        lines += len(re.findall(r"^\s*//", text, flags=re.M))
    return lines, text.count("\n") + 1


def main() -> None:
    per_file = {}
    for d in PY_DIRS:
        for f in sorted((ROOT / d).rglob("*.py")) if (ROOT / d).is_dir() else []:
            if SKIP & set(f.parts):
                continue
            c, doc, n = py_counts(f)
            per_file[str(f.relative_to(ROOT))] = {"comments": c, "docstring_lines": doc, "lines": n}
    for pattern in ("static/js/*.js", "static/css/*.css", "templates/**/*.html"):
        for f in sorted(ROOT.glob(pattern)):
            c, n = web_counts(f)
            per_file[str(f.relative_to(ROOT))] = {"comments": c, "docstring_lines": 0, "lines": n}
    total = {k: sum(v[k] for v in per_file.values()) for k in ("comments", "docstring_lines", "lines")}
    doc = {"total": total, "files": per_file}
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(doc, indent=1))
    print(json.dumps(total))
    worst = sorted(per_file.items(), key=lambda kv: -(kv[1]["comments"] + kv[1]["docstring_lines"]))[:25]
    for name, v in worst:
        share = (v["comments"] + v["docstring_lines"]) / max(v["lines"], 1)
        print(f"{v['comments']:5d} {v['docstring_lines']:5d} {v['lines']:6d} {share:5.0%}  {name}")


if __name__ == "__main__":
    main()
