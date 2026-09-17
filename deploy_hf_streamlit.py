"""Deploy DocuLIFT to HuggingFace Spaces using FREE Gradio SDK."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent
SPACE_NAME = "doculift-invoice-parser"

README = """\
---
title: DocuLIFT Invoice Parser
emoji: ⚡
colorFrom: purple
colorTo: cyan
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: AI invoice and receipt OCR parser
---

# ⚡ DocuLIFT — Intelligent Invoice & Receipt Parser

Upload an invoice or receipt image. DocuLIFT uses PaddleOCR and
rule-based extraction to pull vendor details, dates, amounts, and line items.

> **Note:** First load takes a few minutes while OCR models download.
"""


def copy_tree(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", "*.log", ".DS_Store", "venv"
    )
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def stage(staging: Path) -> None:
    # Core source directories
    copy_tree(ROOT / "src", staging / "src")
    copy_tree(ROOT / "config", staging / "config")

    # Gradio app entry point
    shutil.copy2(ROOT / "app.py", staging / "app.py")

    # Slim requirements (no torch/LayoutLM)
    shutil.copy2(ROOT / "requirements-space.txt", staging / "requirements.txt")

    # pyproject so src package is importable
    shutil.copy2(ROOT / "pyproject.toml", staging / "pyproject.toml")

    # HF Space README with YAML front-matter
    (staging / "README.md").write_text(README, encoding="utf-8")

    # Writable data dirs
    for sub in ("uploads", "outputs", "temp"):
        target = staging / "data" / sub
        target.mkdir(parents=True, exist_ok=True)
        (target / ".gitkeep").write_text("", encoding="utf-8")


def main() -> None:
    api = HfApi()

    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    print(f"Creating / reusing Space: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",        # FREE tier
        private=False,
        exist_ok=True,
    )

    staging = Path(tempfile.mkdtemp(prefix="hfspace-gradio-"))
    try:
        print("Staging files...")
        stage(staging)
        print(f"Uploading from {staging} ...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy DocuLIFT invoice parser (Gradio SDK - free)",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    live_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n✅  LIVE URL = {live_url}")
    print("Space is now building. First build takes ~3-5 minutes.")


if __name__ == "__main__":
    main()
