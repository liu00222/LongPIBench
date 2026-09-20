"""Download and validate the local LongPIBench dataset snapshot."""

from __future__ import annotations

from pathlib import Path

DATASET_REPO_ID = "RainWatcher/LongPIBench"
DATA_ROOT = Path(__file__).resolve().parent / "datasets"
DATASET_DIRECTORIES = ("papers", "person_info", "emails", "code_changes")
PAPER_FILES = (
    "abstract.tex",
    "intro.tex",
    "rw.tex",
    "method.tex",
    "eval.tex",
    "dl.tex",
    "conclusion.tex",
)
ITEM_IDS = range(100)


def missing_dataset_files(root: Path = DATA_ROOT) -> list[Path]:
    """Return required benchmark files that are absent from a local snapshot."""
    missing = []
    for item_id in ITEM_IDS:
        missing.extend(
            root / "papers" / str(item_id) / filename
            for filename in PAPER_FILES
            if not (root / "papers" / str(item_id) / filename).is_file()
        )
        missing.extend(
            root / directory / f"{item_id}.json"
            for directory in ("person_info", "emails", "code_changes")
            if not (root / directory / f"{item_id}.json").is_file()
        )
    return missing


def dataset_error(path: Path) -> FileNotFoundError:
    """Build a consistent error explaining how to install missing dataset files."""
    return FileNotFoundError(
        f"LongPIBench dataset file not found: {path}. "
        "Download the complete dataset with `longpibench download-data`."
    )


def require_dataset_file(path: Path) -> Path:
    """Return a dataset path or raise an actionable missing-data error."""
    if not path.is_file():
        raise dataset_error(path)
    return path


def validate_dataset(root: Path = DATA_ROOT) -> Path:
    """Verify that every required local example is present."""
    root = root.resolve()
    missing = missing_dataset_files(root)
    if missing:
        preview = ", ".join(str(path.relative_to(root)) for path in missing[:5])
        remainder = len(missing) - min(len(missing), 5)
        suffix = f" (and {remainder} more)" if remainder else ""
        raise FileNotFoundError(
            f"LongPIBench dataset is incomplete under {root}; missing: {preview}{suffix}. "
            "Run `longpibench download-data` to download a complete snapshot."
        )
    return root


def download_dataset(
    local_dir: Path = DATA_ROOT,
    *,
    revision: str = "main",
    force_download: bool = False,
) -> Path:
    """Download all four suites from Hugging Face and validate the snapshot."""
    from huggingface_hub import snapshot_download

    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=[f"{directory}/**" for directory in DATASET_DIRECTORIES],
        force_download=force_download,
    )
    return validate_dataset(local_dir)
