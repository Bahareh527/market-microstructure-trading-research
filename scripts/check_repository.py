"""Fail on common accidental disclosures before publication."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".txt", ".yml", ".yaml", ".cff", ".ipynb", ".json", ".csv"
}
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
CHECKS = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.IGNORECASE),
    "private key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "generic secret assignment": re.compile(
        r"(?i)(?:api[_-]?key|secret|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]{8,}"
    ),
}
ALLOWED_PATHS = {Path("scripts/check_repository.py")}


failures: list[str] = []
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
        continue
    relative = path.relative_to(ROOT)
    if any(part in SKIP_PARTS for part in relative.parts) or relative in ALLOWED_PATHS:
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for name, pattern in CHECKS.items():
        if pattern.search(text):
            failures.append(f"{relative}: {name}")
    if path.suffix.lower() == ".ipynb":
        notebook = json.loads(text)
        for cell in notebook.get("cells", []):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    failures.append(f"{relative}: saved notebook error output")

if failures:
    raise SystemExit("Repository safety check failed:\n" + "\n".join(failures))
print("Repository safety check passed.")
