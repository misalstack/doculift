"""Stage and upload this app to a free Hugging Face Space (Docker SDK)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent
SPACE_NAME = "doculift-invoice-parser"

COPY_DIRS = ["src", "config"]
COPY_FILES = [
    "pyproject.toml",
    "run_app.py",
    "requirements-space.txt",
    "Dockerfile",
]

README = """\
---
title: DocuLIFT Invoice Parser
emoji: 🧾
colorFrom: purple
colorTo: cyan
sdk: docker
pinned: false
license: mit
short_description: AI invoice and receipt OCR parser
---

# DocuLIFT Invoice Parser

Upload an invoice or receipt (image or PDF). The app uses PaddleOCR plus
rule-based extraction to pull vendor details, dates, amounts, and line items.

> **Note:** First load can take a few minutes while OCR models download.
"""


def copy_tree(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".pytest_cache", "*.log", ".DS_Store"
    )
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def stage(staging: Path) -> None:
    # Directories
    for name in COPY_DIRS:
        copy_tree(ROOT / name, staging / name)

    # Individual files
    for name in COPY_FILES:
        shutil.copy2(ROOT / name, staging / name)

    # Streamlit theme config
    streamlit_dir = staging / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / ".streamlit" / "config.toml", streamlit_dir / "config.toml")

    # HF Space README (with YAML front-matter)
    (staging / "README.md").write_text(README, encoding="utf-8")

    # Empty data dirs so the container can write to them
    for sub in ("uploads", "outputs", "temp"):
        target = staging / "data" / sub
        target.mkdir(parents=True, exist_ok=True)
        (target / ".gitkeep").write_text("", encoding="utf-8")


def main() -> None:
    api = HfApi()

    # Resolve the logged-in username so the repo_id is deterministic
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    print(f"Creating / reusing Space: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    staging = Path(tempfile.mkdtemp(prefix="hfspace-"))
    try:
        print("Staging files …")
        stage(staging)
        print(f"Uploading from {staging} …")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy DocuLIFT invoice parser (Docker)",
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    live_url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"\n✅  LIVE_URL = {live_url}")
    print("The Space is now building. First build takes ~5-10 minutes.")


if __name__ == "__main__":
    main()
