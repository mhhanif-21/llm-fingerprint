"""Pemeriksaan docstring dan pola yang dilarang pada berkas proyek."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOTS = [ROOT / name for name in ["llmname", "llm_eda", "tests", "scripts"]]


def check() -> list[str]:
    """Mengumpulkan pelanggaran tanpa membaca data atau direktori environment."""
    errors = []
    for directory in CODE_ROOTS:
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text())
            nodes = [
                tree,
                *[
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ],
            ]
            for node in nodes:
                doc = ast.get_docstring(node)
                if not doc or len(doc.splitlines()) > 2:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 1)} docstring wajib 1–2 baris"
                    )
    paths = [p for directory in CODE_ROOTS for p in directory.rglob("*.py")]
    paths += list((ROOT / "docs").rglob("*.md")) + [ROOT / "AGENTS.md", ROOT / "README.md"]
    for path in paths:
        content = path.read_text()
        if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", content):
            errors.append(f"{path.relative_to(ROOT)} mengandung emoji")
        if re.search(r"(?m)^\s*(?:#\s*)?(?:={5,}|-{5,})\s*$", content):
            errors.append(f"{path.relative_to(ROOT)} mengandung pembatas dekoratif")
    return errors


if __name__ == "__main__":
    violations = check()
    print("\n".join(violations) if violations else "Cleanliness: lulus")
    sys.exit(bool(violations))
