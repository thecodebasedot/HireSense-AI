"""
HireSense AI — model registry.

Keeps a versioned history of trained models so you can list past runs, inspect
their metrics, and roll back to any of them. Each registered model is a joblib
bundle stored under models/registry/, tracked in an index.json.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone

import joblib

from config import MODEL_PATH, REGISTRY_DIR

INDEX_PATH = REGISTRY_DIR / "index.json"


def _load_index() -> list[dict]:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return []


def _save_index(index: list[dict]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2))


def register(bundle: dict, make_current: bool = True) -> dict:
    """Store a model bundle in the registry and (optionally) make it current."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    version = bundle.get("version", "0.0.0")
    fname = f"{version}_{ts}.joblib"
    joblib.dump(bundle, REGISTRY_DIR / fname)

    entry = {
        "file": fname,
        "version": version,
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": bundle.get("algorithm"),
        "test_accuracy": bundle.get("test_accuracy"),
        "test_roc_auc": bundle.get("test_roc_auc"),
    }
    index = _load_index()
    index.append(entry)
    _save_index(index)

    if make_current:
        shutil.copy(REGISTRY_DIR / fname, MODEL_PATH)
    return entry


def list_models() -> list[dict]:
    return _load_index()


def rollback(file: str) -> None:
    """Make a previously registered model the current one."""
    src = REGISTRY_DIR / file
    if not src.exists():
        raise FileNotFoundError(f"No registered model '{file}'.")
    shutil.copy(src, MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the model registry.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list", help="List registered models.")
    rb = sub.add_parser("rollback", help="Roll back to a registered model.")
    rb.add_argument("file", help="Registry filename (see `list`).")
    args = parser.parse_args()

    if args.cmd == "rollback":
        rollback(args.file)
        print(f"✓ Rolled back to {args.file} (now current at {MODEL_PATH.name})")
    else:
        models = list_models()
        if not models:
            print("Registry is empty. Train a model to populate it.")
            return
        print(f"{'file':40s} {'acc':>6s} {'auc':>6s}  registered_at")
        for m in models:
            print(f"{m['file']:40s} {m.get('test_accuracy', 0):6.3f} "
                  f"{m.get('test_roc_auc', 0):6.3f}  {m['registered_at']}")


if __name__ == "__main__":
    main()
