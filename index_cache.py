"""Numeric-only cache for a single desktop index snapshot, no pickle."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from importlib.metadata import version
from app_paths import application_root

import numpy as np
from PySide6.QtCore import QIODevice, QSaveFile

MODELS = {"bge-m3", "multilingual-e5-small", "multilingual-e5-base"}
MAX_BYTES = 256 * 1024 * 1024


def cache_signature(model_name, chunks):
    model_dir = application_root() / "models" / model_name
    files = [(p.relative_to(model_dir).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted(model_dir.rglob("*")) if p.is_file()]
    payload = {"version": 1, "model": model_name, "files": files,
               "split": [500, 80], "prefixes": model_name.startswith("multilingual-e5-"),
               "packages": {p: version(p) for p in ["sentence-transformers", "transformers", "langchain-text-splitters"]},
               "chunks": [(c.page_content, c.metadata["source"], c.metadata["start_index"]) for c in chunks]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _read(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Index cache too large.")
    with ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or sum(e.file_size for e in entries) > MAX_BYTES:
            raise ValueError("Invalid cache archive.")
    return np.load(io.BytesIO(data), allow_pickle=False)


def cached_model(path):
    try:
        with _read(path) as stored:
            model = json.loads(str(stored["manifest"].item()))["model"]
            return model if model in MODELS else None
    except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError):
        return None


def load_vectors(path, signature, rows, dimension):
    try:
        with _read(path) as stored:
            manifest = json.loads(str(stored["manifest"].item()))
            if manifest.get("signature") != signature:
                return None
            vectors = stored["vectors"]
            if (vectors.dtype != np.float32 or vectors.shape != (rows, dimension)
                    or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
                return None
            return vectors.copy()
    except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError):
        return None


def save_vectors(path, signature, model_name, vectors):
    data = io.BytesIO()
    np.savez(data, manifest=np.array(json.dumps({"signature": signature, "model": model_name})),
             vectors=vectors.astype(np.float32))
    payload = data.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError("Index cache too large.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = QSaveFile(str(path))
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError("Cannot open index cache.")
    if output.write(payload) != len(payload):
        output.cancelWriting()
        raise OSError("Cannot write index cache.")
    if not output.commit():
        raise OSError("Cannot commit index cache.")
