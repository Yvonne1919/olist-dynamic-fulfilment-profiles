"""Standard-library-only checks of the public inventory and quick-start paths."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "user-home path": re.compile(r"/(?:Users|home)/[^\s\"']+|[A-Za-z]:\\(?:Users|Documents and Settings)\\"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "OpenAI-shaped key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"),
    "credential URL": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
}
ROW_COLUMNS = {"order_id", "customer_id", "customer_unique_id", "review_id", "entity_id", "seller_id", "product_id", "review_comment_message", "review_comment_title"}

def main():
    errors = []
    inventory = ROOT / "RELEASE_FILES.csv"
    if not inventory.is_file():
        raise SystemExit("RELEASE_FILES.csv is missing")
    rows = list(csv.DictReader(inventory.open(encoding="utf-8")))
    known = {row["path"] for row in rows}
    for row in rows:
        p = ROOT / row["path"]
        if p.is_symlink() or not p.is_file() or ROOT not in p.resolve().parents:
            errors.append(f"Missing, symlinked or out-of-root file: {row['path']}")
            continue
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["release_sha256"]:
            errors.append(f"Hash mismatch: {row['path']}")
        if len(data) > 50_000_000:
            errors.append(f"Large file: {row['path']}")
        if p.name.startswith("olist_") and p.name.endswith("_dataset.csv"):
            errors.append(f"Raw dataset: {row['path']}")
        text = data.decode("utf-8")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label}: {row['path']}")
        if p.suffix == ".py":
            try:
                tree = ast.parse(text, filename=row["path"])
                compile(tree, row["path"], "exec")
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("src.", "analysis.")):
                        target = ROOT.joinpath(*node.module.split("."))
                        if not target.is_dir() and not target.with_suffix(".py").is_file():
                            errors.append(f"Missing project import in {row['path']}: {node.module}")
            except SyntaxError as exc:
                errors.append(f"Python syntax: {row['path']}: {exc.msg}")
        elif p.suffix == ".json":
            try:
                json.loads(text)
            except ValueError:
                errors.append(f"Invalid JSON: {row['path']}")
        elif p.suffix == ".csv":
            header = next(csv.reader(text.splitlines()), [])
            if ROW_COLUMNS.intersection(header):
                errors.append(f"Row/entity identifier column: {row['path']}")
            if re.search(r"(?<![a-f0-9])[a-f0-9]{32}(?![a-f0-9])", text):
                errors.append(f"Possible raw identifier in CSV: {row['path']}")
    readme = (ROOT / "README.md").read_text()
    command_count = 0
    for block in re.findall(r"```bash\n(.*?)```", readme, re.S):
        for line in block.splitlines():
            if not line.strip():
                continue
            command_count += 1
            for word in shlex.split(line):
                if word.endswith((".py", ".R")) and "/" in word:
                    if word not in known:
                        errors.append(f"README command points outside release inventory: {word}")
    for name in ("README.md", "REPRODUCIBILITY.md", "RELEASE_MANIFEST.md", "data/README.md"):
        for link in re.findall(r"\]\(([^)]+)\)", (ROOT / name).read_text()):
            if not link.startswith(("https://", "http://", "#")):
                if not (ROOT / Path(name).parent / link.split("#")[0]).exists():
                    errors.append(f"Missing local link in {name}: {link}")
    for message in errors:
        print(message, file=sys.stderr)
    print(json.dumps({"files_checked": len(rows), "readme_commands_checked": command_count,
                      "errors": len(errors), "scan_limit": "pattern checks, not a guarantee against every secret format"}))
    raise SystemExit(bool(errors))

if __name__ == "__main__":
    main()
