"""Read one UTF-8 text file without making a model request."""

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import os
from langchain_core.documents import Document


MAX_IMPORT_BYTES = 1024 * 1024
MAX_MATERIAL_FILES = 500
MAX_MATERIAL_BYTES = 20 * MAX_IMPORT_BYTES
MAX_MATERIAL_TEXT_BYTES = 20 * MAX_IMPORT_BYTES
MAX_FOLDER_ENTRIES = 10000
SUPPORTED_IMPORT_SUFFIXES = {".txt", ".md", ".docx", ".xlsx"}


def import_file_id(path: Path) -> str:
    """Identify the same selected Windows file without persisting its path."""
    return hashlib.sha256(str(path.resolve()).casefold().encode("utf-8")).hexdigest()


def unique_source_name(path: Path, sources: set[str]) -> str:
    if path.name.casefold() not in sources:
        return path.name
    folder = (path.parent.name or "root")[:160]
    for number in range(1, MAX_MATERIAL_FILES + 2):
        label = f" ({folder})" if number == 1 else f" ({folder}, {number})"
        stem = path.stem[:255 - len(label) - len(path.suffix)]
        candidate = stem + label + path.suffix
        if candidate.casefold() not in sources:
            return candidate
    raise ValueError("无法生成唯一的材料名称。")


def collect_import_paths(folder: Path) -> list[Path]:
    """Bound recursive enumeration; never follow descendant links or junctions."""
    root = folder.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("请选择文件夹。")
    pending, paths, scanned = [root], [], 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                scanned += 1
                if scanned > MAX_FOLDER_ENTRIES:
                    raise ValueError(f"文件夹条目超过 {MAX_FOLDER_ENTRIES}，请选择更小的子目录。")
                path = Path(entry.path)
                if path.is_symlink() or path.is_junction():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if not entry.name.startswith("."):
                        pending.append(path)
                elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in SUPPORTED_IMPORT_SUFFIXES:
                    paths.append(path)
                    if len(paths) > MAX_MATERIAL_FILES:
                        raise ValueError(f"文件夹中支持的文件超过 {MAX_MATERIAL_FILES} 份，请分批选择。")
    return sorted(paths)


def _office_text(data: bytes, suffix: str) -> str:
    # OOXML is compressed: bound expansion before handing it to a parser.
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 2000 or sum(e.file_size for e in entries) > 20 * MAX_IMPORT_BYTES:
                raise ValueError("Office 文件解压后过大或条目过多。")
            if any(e.flag_bits & 1 for e in entries):
                raise ValueError("不支持加密 Office 文件。")
            for entry in entries:
                with archive.open(entry) as stream:
                    count = 0
                    while block := stream.read(65536):
                        count += len(block)
                        if count > entry.file_size:
                            raise ValueError("Office 文件解压大小异常。")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Office 文件损坏或格式不匹配，请另存为 DOCX/XLSX。") from exc

    lines, text_bytes = [], 0

    def append(line):
        nonlocal text_bytes
        text_bytes += len(line.encode("utf-8")) + 1
        if text_bytes > MAX_IMPORT_BYTES:
            raise ValueError("解析后的正文不能超过 1 MiB。")
        lines.append(line)

    try:
        if suffix == ".docx":
            from docx import Document as WordDocument
            from docx.text.paragraph import Paragraph

            document = WordDocument(BytesIO(data))
            for index, block in enumerate(document.iter_inner_content(), 1):
                if isinstance(block, Paragraph):
                    if block.text.strip():
                        append(f"[正文段落 {index}] {block.text}")
                else:
                    for row_index, row in enumerate(block.rows, 1):
                        values = [cell.text.replace("\n", " / ") for cell in row.cells]
                        if any(value.strip() for value in values):
                            append(f"[表格 {index} 第 {row_index} 行] " + " | ".join(values))
        else:
            from openpyxl import load_workbook

            workbook = load_workbook(BytesIO(data), read_only=True, data_only=False, keep_links=False)
            try:
                scanned = 0
                for sheet in workbook.worksheets:
                    if (sheet.max_row or 0) > 20000 or (sheet.max_column or 0) > 200:
                        raise ValueError("工作表范围过大，最多 20000 行、200 列。")
                    for row_index, row in enumerate(sheet.iter_rows(), 1):
                        if row_index > 20000 or len(row) > 200:
                            raise ValueError("工作表范围过大，最多 20000 行、200 列。")
                        scanned += len(row)
                        if scanned > 100000:
                            raise ValueError("工作簿扫描单元格数量不能超过 100000。")
                        values = []
                        for cell in row:
                            if cell.value is not None and str(cell.value).strip():
                                value = str(cell.value).replace("\n", " / ")
                                if cell.data_type == "f":
                                    value = f"公式（未计算）: {value}"
                                values.append(f"{cell.coordinate}={value}")
                        if values:
                            append(f"[工作表: {sheet.title} | 第 {row_index} 行] " + " | ".join(values))
            finally:
                workbook.close()
    except ImportError as exc:
        raise ValueError("缺少 Office 解析依赖，请安装 requirements-office.txt。") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Office 文件无法解析，请检查格式；不支持密码保护文件。") from exc
    return "\n".join(lines)


def load_imported_text(path: Path) -> Document:
    """Convert an explicitly selected, bounded file to searchable text."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMPORT_SUFFIXES:
        raise ValueError("仅支持 TXT、MD、DOCX 和 XLSX；旧版 DOC/XLS 请先另存为新格式。")
    with path.open("rb") as file:
        data = file.read(MAX_IMPORT_BYTES + 1)
    if len(data) > MAX_IMPORT_BYTES:
        raise ValueError("单文件不能超过 1 MiB。")
    text = data.decode("utf-8-sig") if suffix in {".txt", ".md"} else _office_text(data, suffix)
    if not text.strip():
        raise ValueError("文件正文为空。")
    return Document(page_content=text, metadata={"source": path.name, "bytes": len(data),
                                                "import_id": import_file_id(path)})


def load_text(path: Path) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"Empty document: {path.name}")

    return Document(
        page_content=text,
        metadata={"source": path.name},
    )


def load_directory(directory: Path) -> list[Document]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    if not paths:
        raise ValueError("No TXT or Markdown files found.")

    documents = []
    for path in paths:
        documents.append(load_text(path))
        print(f"Loaded {path.name}")
    print(f"Total {len(documents)} documents")
    return documents


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "corpus"
    documents = load_directory(data_dir)
    print("Documents:", len(documents))
    for document in documents:
        print("Source:", document.metadata["source"])
        print("Characters:", len(document.page_content))
        print("Preview:", document.page_content[:100])
