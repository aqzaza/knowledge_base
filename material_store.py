"""Bounded material-only JSON storage with atomic Qt writes."""

import json
from pathlib import Path

from PySide6.QtCore import QIODevice, QSaveFile, QStandardPaths
from langchain_core.documents import Document

from read_document import (
    MAX_IMPORT_BYTES, MAX_MATERIAL_FILES, MAX_MATERIAL_BYTES, MAX_MATERIAL_TEXT_BYTES,
    SUPPORTED_IMPORT_SUFFIXES,
)

MAX_STORE_BYTES = 128 * MAX_IMPORT_BYTES


def default_material_path() -> Path:
    root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
    if not root:
        raise OSError("No writable application data location.")
    return Path(root) / "AIResearchAssistant" / "materials.json"


def _validate(payload):
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid material store version.")
    items = payload.get("materials")
    if not isinstance(items, list) or len(items) > MAX_MATERIAL_FILES:
        raise ValueError("Invalid material count.")
    documents, names, identities, total, text_total = [], set(), set(), 0, 0
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Invalid material entry.")
        source, text, size = item.get("source"), item.get("text"), item.get("bytes")
        if (not isinstance(source, str) or not source or len(source) > 255
                or any(c in source for c in '/\\\x00\r\n')
                or Path(source).suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES
                or source.casefold() in names):
            raise ValueError("Invalid or duplicate material name.")
        if (not isinstance(text, str) or not text.strip()
                or len(text.encode("utf-8")) > 3 * MAX_IMPORT_BYTES
                or type(size) is not int or not 0 < size <= MAX_IMPORT_BYTES):
            raise ValueError("Invalid material content or size.")
        identity = item.get("import_id")
        if "import_id" in item:
            if (not isinstance(identity, str) or len(identity) != 64
                    or any(c not in "0123456789abcdef" for c in identity)
                    or identity in identities):
                raise ValueError("Invalid or duplicate import identity.")
            identities.add(identity)
        total += size
        text_total += len(text.encode("utf-8"))
        if total > MAX_MATERIAL_BYTES or text_total > MAX_MATERIAL_TEXT_BYTES:
            raise ValueError("Material collection exceeds size limit.")
        names.add(source.casefold())
        metadata = {"source": source, "bytes": size}
        if identity is not None:
            metadata["import_id"] = identity
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def load_materials(path: Path) -> list[Document]:
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_STORE_BYTES + 1)
    except FileNotFoundError:
        return []
    if len(data) > MAX_STORE_BYTES:
        raise ValueError("Material store is too large.")
    return _validate(json.loads(data.decode("utf-8")))


def save_materials(path: Path, documents: list[Document]) -> None:
    payload = {"version": 1, "materials": [
        {"source": d.metadata["source"], "bytes": d.metadata["bytes"], "text": d.page_content,
         **({"import_id": d.metadata["import_id"]} if "import_id" in d.metadata else {})}
        for d in documents
    ]}
    _validate(payload)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(data) > MAX_STORE_BYTES:
        raise ValueError("Material store is too large.")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = QSaveFile(str(path))
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError("Cannot open material store for writing.")
    if output.write(data) != len(data):
        output.cancelWriting()
        raise OSError("Cannot write material store.")
    if not output.commit():
        raise OSError("Cannot commit material store.")
