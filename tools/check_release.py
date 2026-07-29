"""Dependency-free checks for the reproducibility package."""

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist"}
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {".ckpt", ".h5", ".onnx", ".pb", ".pt", ".pth", ".safetensors"}
MAX_FILE_BYTES = 50 * 1024 * 1024
REQUIRED_FILES = {
    ".gitattributes",
    ".gitignore",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CONTRIBUTING.zh-CN.md",
    "LICENSE",
    "QUICKSTART.md",
    "QUICKSTART.zh-CN.md",
    "README.md",
    "README.zh-CN.md",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_NOTICES.zh-CN.md",
    "docs/EXPERIMENTS.md",
    "docs/EXPERIMENTS.zh-CN.md",
    "environment.yml",
    "experiments/run_multi_source.py",
    "requirements.txt",
}
SUPPORTED_ATTACKS = ("l2t", "bsr", "decowa", "ops", "sid", "eda")
TRANSFERATTACK_PY_FILES = {
    "__init__.py",
    "_momentum.py",
    "attack.py",
    "utils.py",
    "input_transformation/__init__.py",
    *(f"input_transformation/{name}.py" for name in SUPPORTED_ATTACKS),
}
SENSITIVE_PATTERNS = {
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    "Windows user path": re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+"),
    "local Linux user path": re.compile(r"/(?:root|home/[^/\s]+)/"),
}
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def iter_release_files():
    for path in ROOT.rglob("*"):
        relative_parts = path.relative_to(ROOT).parts
        if not path.is_file() or any(part in SKIP_PARTS for part in relative_parts):
            continue
        yield path


def relative(path):
    return path.relative_to(ROOT).as_posix()


def check_required(errors):
    for name in sorted(REQUIRED_FILES):
        if not (ROOT / name).is_file():
            errors.append(f"missing required file: {name}")


def check_files(errors):
    for path in iter_release_files():
        rel = relative(path)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden model/checkpoint file: {rel}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 50 MiB: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".env.example",
            ".gitattributes",
            ".gitignore",
            "LICENSE",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{label} in {rel}:{line}")


def check_markdown_links(errors):
    for path in iter_release_files():
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            if "://" in target or target.startswith("#"):
                continue
            clean_target = target.split("#", 1)[0]
            if not clean_target:
                continue
            if not (path.parent / clean_target).resolve().exists():
                errors.append(f"broken local link in {relative(path)}: {target}")


def check_attack_scope(errors):
    package = ROOT / "transferattack"
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    for path in sorted(TRANSFERATTACK_PY_FILES - actual):
        errors.append(f"missing transferattack support file: {path}")
    for path in sorted(actual - TRANSFERATTACK_PY_FILES):
        errors.append(f"unexpected transferattack method or module: {path}")

    registry_path = package / "__init__.py"
    try:
        tree = ast.parse(registry_path.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "attack_zoo"
                for target in node.targets
            )
        )
        registry = ast.literal_eval(assignment.value)
    except (OSError, SyntaxError, StopIteration, ValueError) as exc:
        errors.append(f"cannot read attack registry: {exc}")
        return
    if tuple(registry) != SUPPORTED_ATTACKS:
        errors.append(
            "attack registry must contain exactly: " + ", ".join(SUPPORTED_ATTACKS)
        )


def main():
    errors = []
    check_required(errors)
    check_files(errors)
    check_markdown_links(errors)
    check_attack_scope(errors)
    if errors:
        print("Release check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    count = sum(1 for _ in iter_release_files())
    print(f"Release check passed for {count} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
