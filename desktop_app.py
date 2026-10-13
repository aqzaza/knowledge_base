"""Desktop local retrieval with a selectable embedding model."""

import sys
import json
import io
import re
import uuid
from pathlib import Path
from material_store import default_material_path, load_materials, save_materials
from local_answer import LocalAnswerSession
from read_document import MAX_MATERIAL_FILES, MAX_MATERIAL_BYTES, MAX_MATERIAL_TEXT_BYTES

from PySide6.QtCore import QThread, Signal, Qt, QUrl
from PySide6.QtGui import QTextDocument, QImage, QTextImageFormat
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QMainWindow,
    QStyle,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class EvidenceDocument(QTextDocument):
    def loadResource(self, resource_type, url):
        # Imported evidence must not fetch remote images or read local files.
        return None

    def set_reading_markdown(self, text):
        """Render bounded math locally; leave code and unsupported formulas intact."""
        from matplotlib import rc_context
        from matplotlib.mathtext import math_to_image
        from matplotlib.font_manager import FontProperties

        pattern = re.compile(
            r"(?P<code>^```[^\n]*\n.*?^```[^\n]*$|^~~~[^\n]*\n.*?^~~~[^\n]*$|`+[^`\n]*`+)"
            r"|(?P<math>(?<!\\)\$\$.+?(?<!\\)\$\$|\\\[.+?\\\]|\\\(.+?\\\)"
            r"|(?<![\\$])\$(?!\$)[^\n$]+?(?<!\\)\$(?!\$))",
            re.MULTILINE | re.DOTALL,
        )
        images = []
        prefix = "formula" + uuid.uuid4().hex

        def replace(match):
            original = match.group(0)
            if match.group("code") or len(images) >= 32 or len(original) > 300:
                return original
            body = original[2:-2] if original.startswith(("$$", "\\[", "\\(")) else original[1:-1]
            try:
                stream = io.BytesIO()
                with rc_context({"text.usetex": False, "mathtext.fontset": "dejavusans"}):
                    math_to_image("$" + body.strip() + "$", stream, format="png",
                                  prop=FontProperties(size=14), dpi=144, color="#163247")
                image = QImage.fromData(stream.getvalue(), "PNG")
                if image.isNull() or image.width() > 2400 or image.height() > 600:
                    return original
            except (ValueError, RuntimeError):
                return original
            marker = prefix + str(len(images)) + "end"
            images.append((marker, image))
            return marker

        self.setMarkdown(pattern.sub(replace, text),
                         QTextDocument.MarkdownFeature.MarkdownDialectGitHub
                         | QTextDocument.MarkdownFeature.MarkdownNoHTML)
        for marker, image in images:
            cursor = self.find(marker)
            if cursor.isNull():
                continue
            url = QUrl("formula:" + marker)
            self.addResource(QTextDocument.ResourceType.ImageResource, url, image)
            formatting = QTextImageFormat()
            formatting.setName(url.toString())
            formatting.setWidth(image.width() / 2)
            formatting.setHeight(image.height() / 2)
            formatting.setVerticalAlignment(QTextImageFormat.VerticalAlignment.AlignMiddle)
            cursor.insertImage(formatting)


class RetrievalWorker(QThread):
    succeeded = Signal(object, object)
    failed = Signal(str)
    answered = Signal(object)
    answer_failed = Signal(str)
    progress = Signal(str)

    def __init__(self, model_name, question, index, parent=None, generate=False, documents=None,
                 api_config=None, local=False, local_device="auto", local_session=None):
        super().__init__(parent)
        self.model_name = model_name
        self.question = question
        self.index = index
        self.generate = generate
        self.local = local
        self.local_device = local_device
        self.local_session = local_session
        self.documents = documents
        self.api_config = dict(api_config) if api_config else None

    def run(self):
        try:
            # Import the ML stack off the UI thread as well as running inference.
            from embed_query import load_index
            from langchain_core.documents import Document
            from document_tools import make_document_tools

            if self.index is None:
                self.index = (load_index(self.model_name) if self.documents is None
                            else load_index(self.model_name, documents=self.documents))
            if self.generate:
                from document_tools import run_agent

                self.progress.emit("DeepSeek 正在选择检索工具……")
                agent_result = run_agent(self.question, self.index, max_tool_calls=3,
                                         api_config=self.api_config)
                self.succeeded.emit(self.index, agent_result["hits"])
                self.answered.emit(agent_result)
                return
            search_documents, read_passage = make_document_tools(self.index)
            search_result = search_documents.invoke({"question": self.question, "top_k": 3})
            hits = []
            for result in search_result["results"]:
                passage = read_passage.invoke({"evidence_id": result["evidence_id"]})
                chunk_number = int(result["evidence_id"].split("-", 1)[1]) - 1
                chunk = Document(
                    page_content=passage["text"],
                    metadata={"source": passage["source"], "start_index": passage["start_index"]},
                )
                hits.append((chunk_number, chunk, result["score"]))
            self.succeeded.emit(self.index, hits)
        except Exception as exc:
            self.index = None
            from embed_query import LocalModelError
            message = str(exc) if isinstance(exc, LocalModelError) else f"检索失败：{type(exc).__name__}。请检查本地模型、可用内存及 API 设置。"
            self.failed.emit(message)
            return
        if self.local:
            try:
                from local_answer import answer_local
                self.progress.emit("本地 Qwen3 正在回答……")
                self.answered.emit(answer_local(self.question, [c for _, c, _ in hits],
                                                device=self.local_device, session=self.local_session))
            except Exception as exc:
                from local_answer import LocalAnswerError
                message = str(exc) if isinstance(exc, LocalAnswerError) else f"本地回答失败：{type(exc).__name__}；检索证据保留。"
                self.answer_failed.emit(message)


class IndexWorker(QThread):
    succeeded = Signal(object, object)
    failed = Signal(str)

    def __init__(self, model_name, documents, parent=None, cache_path=None):
        super().__init__(parent)
        self.model_name = model_name
        self.documents = list(documents)
        self.cache_path = cache_path

    def run(self):
        try:
            from embed_query import load_index

            kwargs = {"documents": self.documents}
            if self.cache_path is not None:
                kwargs["cache_path"] = self.cache_path
            index = load_index(self.model_name, **kwargs)
            self.succeeded.emit(index, self.documents)
        except Exception as exc:
            from embed_query import LocalModelError
            message = str(exc) if isinstance(exc, LocalModelError) else f"材料库更新失败：{type(exc).__name__}。请检查可用内存与模型文件。"
            self.failed.emit(message + " 原知识库保留，未自动重试。")


class MainWindow(QMainWindow):
    def __init__(self, storage_path=None):
        super().__init__()
        self.storage_path = Path(storage_path) if storage_path is not None else default_material_path()
        self.storage_blocked = False
        self.cache_path = self.storage_path.with_name("index.npz")
        self.setWindowTitle("AI 技术资料研究助手")
        self.resize(1100, 760)
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(12)
        self.setStyleSheet("""
            QWidget { font-family: 'Microsoft YaHei UI'; font-size: 14px;
                color: #202c34; background: #f5f7f8; }
            QLabel#heading { font-size: 22px; font-weight: 600; }
            QLabel#metadata { color: #556873; }
            QTextEdit, QLineEdit, QComboBox, QListWidget { background: #ffffff;
                border: 1px solid #ccd6dc; border-radius: 6px; padding: 10px;
                selection-background-color: #d8eeeb; selection-color: #153d38; }
            QTextEdit:focus, QComboBox:focus { border: 1px solid #168576; }
            QListWidget::item { padding: 12px 8px; border-bottom: 1px solid #e8edef; }
            QListWidget::item:selected { background: #d8eeeb; color: #153d38; }
            QPushButton { background: #16786c; color: white; border: none;
                border-radius: 6px; padding: 10px 24px; font-weight: 600; }
            QPushButton:hover { background: #116358; }
            QPushButton:disabled { background: #d1dcda; color: #566b67; }
            QStatusBar { color: #556873; font-size: 12px; }
            QSplitter::handle { background: #e1e7ea; }
        """)
        heading = QLabel("AI 技术资料研究助手")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        self.index = None
        self.index_model = None
        self.worker = None
        self.local_session = LocalAnswerSession()
        self.hits = []
        self.imported_documents = []
        self.active_documents = []
        self.materials_dirty = True

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("检索模型"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["bge-m3", "multilingual-e5-small", "multilingual-e5-base"])
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        toolbar.addWidget(self.model_selector)
        toolbar.addStretch()
        self.import_button = QPushButton("导入资料")
        self.import_button.clicked.connect(self.on_import)
        toolbar.addWidget(self.import_button)
        layout.addLayout(toolbar)
        self.active_source = QLabel("当前知识库：暂无材料")
        self.active_source.setWordWrap(True)
        layout.addWidget(self.active_source)

        layout.addWidget(QLabel("问题"))
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("请输入问题")
        self.question_input.setFixedHeight(90)
        self.question_input.setAcceptRichText(False)
        layout.addWidget(self.question_input)

        self.query_button = QPushButton("查询")
        self.query_button.clicked.connect(self.on_query)
        actions = QHBoxLayout()
        self.answer_mode = QComboBox()
        self.answer_mode.addItems(["仅检索", "本地回答（Qwen3）", "DeepSeek（付费）"])
        actions.addWidget(QLabel("回答模式"))
        actions.addWidget(self.answer_mode)
        self.local_device_selector = QComboBox()
        self.local_device_selector.addItem("自动（优先 GPU）", "auto")
        self.local_device_selector.addItem("GPU（CUDA 13）", "cuda")
        self.local_device_selector.addItem("CPU", "cpu")
        self.local_device_selector.setEnabled(False)
        self.local_device_selector.currentIndexChanged.connect(self.on_local_device_changed)
        actions.addWidget(QLabel("推理设备"))
        actions.addWidget(self.local_device_selector)
        # Compatibility for existing callers; the visible control is the mode menu.
        self.generate_checkbox = QCheckBox()
        self.generate_checkbox.toggled.connect(
            lambda checked: self.answer_mode.setCurrentIndex(2 if checked else 0))
        self.answer_mode.currentIndexChanged.connect(self.on_answer_mode_changed)
        actions.addStretch()
        actions.addWidget(self.query_button)
        layout.addLayout(actions)

        self.result_tabs = QTabWidget()
        answer_panel = QWidget()
        answer_layout = QVBoxLayout(answer_panel)
        self.answer_view = QTextEdit()
        self.answer_view.setReadOnly(True)
        self.answer_view.setDocument(EvidenceDocument(self.answer_view))
        self.citation_status = QLabel()
        self.citation_status.setWordWrap(True)
        self.call_details = QTextEdit()
        self.call_details.setReadOnly(True)
        self.call_details.setMaximumHeight(100)
        answer_layout.addWidget(self.answer_view, 1)
        answer_layout.addWidget(self.citation_status)
        answer_layout.addWidget(QLabel("调用记录"))
        answer_layout.addWidget(self.call_details)
        self.result_tabs.addTab(answer_panel, "回答")
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.evidence_list = QListWidget()
        self.evidence_list.setMinimumWidth(200)
        self.evidence_list.currentRowChanged.connect(self.show_evidence)
        splitter.addWidget(self.evidence_list)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(12, 0, 0, 0)
        self.evidence_metadata = QLabel("原文")
        self.evidence_metadata.setObjectName("metadata")
        self.evidence_metadata.setWordWrap(True)
        detail_layout.addWidget(self.evidence_metadata)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setDocument(EvidenceDocument(self.preview))
        self.evidence_tabs = QTabWidget()
        self.evidence_tabs.addTab(self.preview, "阅读视图")
        self.evidence_tabs.addTab(self.output, "原始文本")
        detail_layout.addWidget(self.evidence_tabs)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 720])
        self.result_tabs.addTab(splitter, "检索证据")
        self.result_tabs.setCurrentIndex(1)
        import_panel = QWidget()
        import_layout = QVBoxLayout(import_panel)
        self.import_status = QLabel("暂无材料")
        self.import_status.setWordWrap(True)
        import_layout.addWidget(self.import_status)
        self.storage_status = QLabel(f"本地材料：{self.storage_path}")
        self.storage_status.setWordWrap(True)
        import_layout.addWidget(self.storage_status)
        management_actions = QHBoxLayout()
        self.import_folder_button = QPushButton("导入文件夹")
        self.import_folder_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.import_folder_button.setToolTip("递归导入 TXT、MD、DOCX、XLSX；不跟随目录链接")
        self.import_folder_button.clicked.connect(self.on_import_folder)
        management_actions.addWidget(self.import_folder_button)
        self.remove_button = QPushButton("移除选中材料")
        self.remove_button.clicked.connect(self.on_remove_material)
        management_actions.addWidget(self.remove_button)
        self.clear_materials_button = QPushButton("清空材料")
        self.clear_materials_button.clicked.connect(self.on_clear_materials)
        management_actions.addWidget(self.clear_materials_button)
        self.rebuild_button = QPushButton("更新材料库")
        self.rebuild_button.setToolTip("将当前材料用于检索；已有匹配缓存时直接复用")
        self.rebuild_button.setEnabled(False)
        self.rebuild_button.clicked.connect(self.on_rebuild)
        management_actions.addWidget(self.rebuild_button)
        import_layout.addLayout(management_actions)
        self.import_list = QListWidget()
        self.import_list.setMinimumHeight(140)
        self.import_list.setMaximumHeight(190)
        self.import_list.currentRowChanged.connect(self.show_imported_document)
        import_layout.addWidget(self.import_list)
        self.import_preview = QTextEdit()
        self.import_preview.setReadOnly(True)
        import_layout.addWidget(self.import_preview, 1)
        self.import_errors = QTextEdit()
        self.import_errors.setReadOnly(True)
        self.import_errors.setMaximumHeight(85)
        import_layout.addWidget(self.import_errors)
        self.result_tabs.addTab(import_panel, "材料管理")
        api_panel = QWidget()
        api_layout = QVBoxLayout(api_panel)
        api_form = QFormLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_model_input = QLineEdit("deepseek-flash")
        self.api_base_input = QLineEdit("https://api.deepseek.com")
        api_form.addRow("API Key", self.api_key_input)
        api_form.addRow("模型", self.api_model_input)
        api_form.addRow("接口地址", self.api_base_input)
        api_layout.addLayout(api_form)
        self.api_status = QLabel("密钥未填写 · 仅本次会话")
        self.api_status.setWordWrap(True)
        api_layout.addWidget(self.api_status)
        self.api_key_input.textChanged.connect(
            lambda text: self.api_status.setText(
                "密钥已填写 · 仅本次会话" if text.strip() else "密钥未填写 · 仅本次会话"))
        self.clear_api_button = QPushButton("清除密钥")
        self.clear_api_button.clicked.connect(self.api_key_input.clear)
        api_layout.addWidget(self.clear_api_button)
        api_layout.addStretch()
        self.api_tab_index = self.result_tabs.addTab(api_panel, "API 设置")
        layout.addWidget(self.result_tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("暂无材料")
        self.set_busy(False)
        try:
            self.imported_documents = load_materials(self.storage_path)
            if self.imported_documents:
                self.mark_materials_dirty()
                self.refresh_materials()
                self.result_tabs.setCurrentIndex(2)
                if self.cache_path.exists():
                    from index_cache import cached_model
                    model_name = cached_model(self.cache_path)
                    if model_name:
                        self.model_selector.setCurrentText(model_name)
                    self.on_rebuild()
        except (OSError, ValueError, UnicodeError):
            self.storage_blocked = True
            self.storage_status.setText("材料恢复失败；原保存文件未修改。请备份并检查该文件：" + str(self.storage_path))
            self.result_tabs.setCurrentIndex(2)

    def persist_materials(self, documents):
        if self.storage_blocked:
            self.storage_status.setText("保存已暂停：请先备份并修复或移走损坏的材料文件，然后重启。路径：" + str(self.storage_path))
            return False
        try:
            save_materials(self.storage_path, documents)
        except (OSError, ValueError):
            self.storage_status.setText("材料保存失败，本次修改未生效；原材料保留。路径：" + str(self.storage_path))
            return False
        self.storage_status.setText(f"材料已保存到本地：{self.storage_path}")
        try:
            self.cache_path.unlink(missing_ok=True)
        except OSError:
            self.storage_status.setText("材料已保存，但旧索引缓存未能删除；恢复时会校验材料是否匹配。")
        return True

    def on_import(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择资料", "", "支持的资料 (*.txt *.md *.docx *.xlsx);;Word (*.docx);;Excel (*.xlsx);;文本 (*.txt *.md)"
        )
        if paths:
            self.import_files(paths)

    def on_import_folder(self):
        if self.worker is not None:
            return
        folder = QFileDialog.getExistingDirectory(self, "选择资料文件夹")
        if folder:
            self.import_folder(Path(folder))

    def import_folder(self, folder):
        from read_document import collect_import_paths

        if self.worker is not None:
            return
        try:
            paths = collect_import_paths(Path(folder))
            if not paths:
                raise ValueError("文件夹中没有支持的 TXT、MD、DOCX 或 XLSX 文件。")
        except ValueError as exc:
            self.import_errors.setPlainText(str(exc))
            self.result_tabs.setCurrentIndex(2)
            return
        except OSError:
            self.import_errors.setPlainText("文件夹无法读取；原材料保留。")
            self.result_tabs.setCurrentIndex(2)
            return
        self.import_files(paths)

    def on_answer_mode_changed(self, mode):
        if mode != 1 and self.worker is None:
            self.local_session.stop()
        self.generate_checkbox.blockSignals(True)
        self.generate_checkbox.setChecked(mode == 2)
        self.generate_checkbox.blockSignals(False)
        self.local_device_selector.setEnabled(mode == 1 and self.worker is None)
        self.clear_answer()

    def on_local_device_changed(self, _):
        if self.worker is None:
            self.local_session.stop()
            self.clear_answer()
            self.statusBar().showMessage("推理设备已切换，下次本地提问将重新加载模型。")

    def import_files(self, paths):
        from read_document import load_imported_text, import_file_id, unique_source_name

        if self.worker is not None:
            return
        self.result_tabs.setCurrentIndex(2)
        if len(paths) > MAX_MATERIAL_FILES:
            self.import_errors.setPlainText(f"每批最多选择 {MAX_MATERIAL_FILES} 个文件；原材料已保留。")
            return
        documents, errors = list(self.imported_documents), []
        original_count = len(documents)
        total_bytes = sum(d.metadata["bytes"] for d in documents)
        text_bytes = sum(len(d.page_content.encode("utf-8")) for d in documents)
        identities = {d.metadata["import_id"] for d in documents if "import_id" in d.metadata}
        sources = {d.metadata["source"].casefold() for d in documents}
        for name in paths:
            path = Path(name)
            try:
                if import_file_id(path) in identities:
                    raise ValueError("该文件已导入，请先移除旧材料再重新导入。")
                document = load_imported_text(path)
                if any("import_id" not in d.metadata and d.metadata["source"].casefold() == path.name.casefold()
                       and d.page_content == document.page_content for d in documents):
                    raise ValueError("旧材料库已有同名且正文相同的材料，请先移除旧材料再重新导入。")
                if len(documents) >= MAX_MATERIAL_FILES:
                    raise ValueError(f"材料库最多容纳 {MAX_MATERIAL_FILES} 个文件。")
                if total_bytes + document.metadata["bytes"] > MAX_MATERIAL_BYTES:
                    raise ValueError(f"材料库总大小不能超过 {MAX_MATERIAL_BYTES // (1024 * 1024)} MiB。")
                added_text_bytes = len(document.page_content.encode("utf-8"))
                if text_bytes + added_text_bytes > MAX_MATERIAL_TEXT_BYTES:
                    raise ValueError(f"材料正文总量不能超过 {MAX_MATERIAL_TEXT_BYTES // (1024 * 1024)} MiB。")
                document.metadata["source"] = unique_source_name(path, sources)
                documents.append(document)
                total_bytes += document.metadata["bytes"]
                text_bytes += added_text_bytes
                identities.add(document.metadata["import_id"])
                sources.add(document.metadata["source"].casefold())
            except UnicodeDecodeError:
                errors.append(f"{path.name}：不是有效 UTF-8 文件，请转换编码。")
            except ValueError as exc:
                errors.append(f"{path.name}：{exc}")
            except OSError:
                errors.append(f"{path.name}：文件无法读取。")
        if len(documents) != original_count and not self.persist_materials(documents):
            self.import_errors.setPlainText("\n".join(errors))
            return
        self.imported_documents = documents
        if len(documents) != original_count:
            self.mark_materials_dirty()
        self.refresh_materials()
        self.import_errors.setPlainText("\n".join(errors) or "无读取错误。")
        self.result_tabs.setCurrentIndex(2)

    def refresh_materials(self):
        self.import_list.clear()
        self.import_preview.clear()
        for document in self.imported_documents:
            self.import_list.addItem(f"{document.metadata['source']} · {len(document.page_content)} 字符")
        if self.imported_documents:
            self.import_list.setCurrentRow(0)
        self.set_busy(False)

    def mark_materials_dirty(self):
        self.local_session.stop()
        self.materials_dirty = True
        self.clear_evidence()
        count = len(self.imported_documents)
        message = f"材料 {count} 份 · 待更新材料库，查询已暂停" if count else "暂无材料"
        self.active_source.setText(message)
        self.import_status.setText(message)
        self.statusBar().showMessage(message)
        if not count:
            self.index = None
            self.index_model = None
            self.active_documents = []

    def on_remove_material(self):
        row = self.import_list.currentRow()
        if self.worker is not None or not 0 <= row < len(self.imported_documents):
            return
        name = self.imported_documents[row].metadata["source"]
        if QMessageBox.question(self, "移除材料", f"从材料库移除 {name}？原文件不会被删除。") == QMessageBox.StandardButton.Yes:
            self.remove_material(row)

    def remove_material(self, row):
        if self.worker is not None or not 0 <= row < len(self.imported_documents):
            return
        documents = list(self.imported_documents)
        del documents[row]
        if not self.persist_materials(documents):
            return
        self.imported_documents = documents
        self.mark_materials_dirty()
        self.refresh_materials()
        self.result_tabs.setCurrentIndex(2)

    def on_clear_materials(self):
        if self.worker is not None or not self.imported_documents:
            return
        if QMessageBox.question(self, "清空材料", "清空材料库？原文件不会被删除。") == QMessageBox.StandardButton.Yes:
            self.clear_materials()

    def clear_materials(self):
        if self.worker is not None:
            return
        if not self.persist_materials([]):
            return
        self.imported_documents = []
        self.mark_materials_dirty()
        self.refresh_materials()
        self.import_errors.clear()
        self.result_tabs.setCurrentIndex(2)

    def show_imported_document(self, row):
        if 0 <= row < len(self.imported_documents):
            self.import_preview.setPlainText(self.imported_documents[row].page_content)

    def set_busy(self, busy):
        for widget in (self.query_button, self.model_selector, self.import_button, self.import_folder_button,
                       self.generate_checkbox, self.answer_mode):
            widget.setEnabled(not busy)
        self.rebuild_button.setEnabled(not busy and bool(self.imported_documents))
        self.remove_button.setEnabled(not busy and bool(self.imported_documents))
        self.clear_materials_button.setEnabled(not busy and bool(self.imported_documents))
        self.query_button.setEnabled(not busy and not self.materials_dirty and self.index is not None)
        self.question_input.setReadOnly(busy)
        self.local_device_selector.setEnabled(not busy and self.answer_mode.currentIndex() == 1)
        for widget in (self.api_key_input, self.api_model_input, self.api_base_input, self.clear_api_button):
            widget.setEnabled(not busy)

    def on_rebuild(self):
        if self.worker is not None or not self.imported_documents:
            return
        self.local_session.stop()
        self.set_busy(True)
        self.statusBar().showMessage("正在加载检索模型并准备材料库……")
        self.worker = IndexWorker(self.model_selector.currentText(), self.imported_documents, self,
                                  cache_path=self.cache_path)
        self.worker.succeeded.connect(self.on_index_ready)
        self.worker.failed.connect(self.on_index_failure)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_index_ready(self, index, documents):
        self.index = index
        self.index_model = self.model_selector.currentText()
        self.active_documents = documents
        self.materials_dirty = False
        self.clear_evidence()
        names = "、".join(d.metadata["source"] for d in documents[:3])
        if len(documents) > 3:
            names += " 等"
        self.active_source.setText(f"当前知识库：导入 {len(documents)} 份资料 · {names}")
        self.import_status.setText(f"索引已就绪：{len(documents)} 份资料，{len(index[1])} 个片段。")
        cache_status = getattr(index[0], "_project_index_cache_status", None)
        if cache_status == "restored":
            self.import_status.setText(self.import_status.text() + " · 已恢复缓存，未重新编码")
        elif cache_status == "save_failed":
            self.storage_status.setText("材料已保存，但索引缓存保存失败；重启后需要重新构建。")
        device = getattr(index[0], "device", None)
        if device is not None:
            self.import_status.setText(self.import_status.text() + f" · 运行设备：{str(device).upper()}")
        self.statusBar().showMessage("索引已就绪，可以查询导入资料。")
        if self.worker is None:
            self.set_busy(False)

    def on_index_failure(self, message):
        self.import_errors.setPlainText(message)
        self.result_tabs.setCurrentIndex(2)
        self.statusBar().showMessage(message)

    def on_model_changed(self, model_name):
        self.local_session.stop()
        self.index = None
        self.index_model = None
        self.clear_evidence()
        self.mark_materials_dirty()
        self.set_busy(False)
        self.statusBar().showMessage(f"已选择 {model_name}，请更新材料库。")

    def on_query(self):
        if self.worker is not None:
            return
        question = self.question_input.toPlainText().strip()
        if not question:
            self.set_message("请输入问题。")
            return
        if self.materials_dirty or self.index is None or not self.active_documents:
            self.set_message("请先在材料管理页导入资料并更新材料库。")
            self.result_tabs.setCurrentIndex(2)
            return
        model_name = self.model_selector.currentText()
        api_config = None
        if self.generate_checkbox.isChecked():
            from document_tools import validate_api_config
            try:
                api_config = validate_api_config({
                    "api_key": self.api_key_input.text(),
                    "model": self.api_model_input.text(),
                    "base_url": self.api_base_input.text(),
                })
            except ValueError as exc:
                self.api_status.setText(str(exc))
                self.result_tabs.setCurrentIndex(self.api_tab_index)
                return
        self.set_busy(True)
        self.clear_answer()
        self.clear_evidence()
        self.statusBar().showMessage(f"正在使用 {model_name} 加载或检索……")
        self.worker = RetrievalWorker(
            model_name, question, self.index, self,
            generate=self.generate_checkbox.isChecked(),
            documents=self.active_documents,
            api_config=api_config,
            local=self.answer_mode.currentIndex() == 1,
            local_device=self.local_device_selector.currentData(),
            local_session=self.local_session,
        )
        self.worker.succeeded.connect(self.on_results)
        self.worker.failed.connect(self.on_failure)
        self.worker.answered.connect(self.on_answer)
        self.worker.answer_failed.connect(self.on_answer_failure)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_results(self, index, hits):
        self.index = index
        self.index_model = self.model_selector.currentText()
        self.clear_evidence()
        self.hits = hits
        for rank, (number, chunk, score) in enumerate(hits, start=1):
            self.evidence_list.addItem(
                f"[{rank}] {chunk.metadata['source']}\n相似度 {score:.4f}"
            )
        if hits:
            self.evidence_list.setCurrentRow(0)
        else:
            self.set_message("没有检索结果。")
        self.statusBar().showMessage(f"检索完成 · {self.index_model} · {len(hits)} 条证据 · 未调用 DeepSeek")

    def clear_evidence(self):
        self.hits = []
        self.evidence_list.clear()
        self.evidence_metadata.setText("原文")
        self.output.clear()
        self.preview.clear()
        self.clear_answer()

    def clear_answer(self):
        self.answer_view.clear()
        self.citation_status.clear()
        self.call_details.clear()
        self.result_tabs.setCurrentIndex(1)

    def on_answer(self, answer):
        self.answer_view.document().setMarkdown(
            answer["answer"],
            QTextDocument.MarkdownFeature.MarkdownDialectGitHub
            | QTextDocument.MarkdownFeature.MarkdownNoHTML,
        )
        if answer["invalid_citations"]:
            message = f"存在无效引用编号：{answer['invalid_citations']}"
        elif not answer["citations"]:
            message = "回答没有引用，请核对是否为合理拒答。"
        else:
            message = f"引用编号有效：{answer['citations']}。证据支持关系仍需人工核对。"
        self.citation_status.setText(message)
        usage = json.dumps(answer["usage"], ensure_ascii=False, indent=2) if answer["usage"] is not None else "未返回 token 用量。"
        trace = answer.get("trace", [])
        trace_text = json.dumps(trace, ensure_ascii=False, indent=2) if trace else "本次未记录工具调用。"
        self.call_details.setPlainText(
            f"返回模型：{answer['model']}\n{usage}\n\n工具调用记录：\n{trace_text}"
        )
        if answer.get("mode") == "local":
            self.call_details.setPlainText(
                f"本地模型：{answer['model']}\n实际推理设备：{answer.get('device', '未知')}\n"
                f"GPU 加载层数：{answer.get('gpu_layers', 0)}\n未调用云端 API。"
            )
        self.result_tabs.setCurrentIndex(0)
        self.statusBar().showMessage(f"回答完成 · 本地 Qwen3 · {answer.get('device', '未知')} · 未调用 API" if answer.get("mode") == "local"
                                     else "回答完成 · DeepSeek · 工具调用 " + str(len(trace)) + " 次")

    def on_answer_failure(self, message):
        self.answer_view.setPlainText(message)
        self.result_tabs.setCurrentIndex(0)
        self.statusBar().showMessage("生成失败 · 检索证据已保留")

    def set_message(self, message):
        self.output.setPlainText(message)
        self.preview.setPlainText(message)

    def show_evidence(self, row):
        if not 0 <= row < len(self.hits):
            return
        number, chunk, score = self.hits[row]
        self.evidence_metadata.setText(
            f"[{row + 1}] {chunk.metadata['source']} · 片段 {number} · "
            f"字符起点 {chunk.metadata['start_index']} · 相似度 {score:.4f}"
        )
        self.output.setPlainText(chunk.page_content)
        self.preview.document().set_reading_markdown(chunk.page_content)

    def on_failure(self, message):
        self.index = None
        self.index_model = None
        self.set_message(message)
        self.statusBar().showMessage("检索失败，未自动重试。")

    def on_finished(self):
        if isinstance(self.worker, RetrievalWorker):
            self.worker.api_config = None
        self.worker.deleteLater()
        self.worker = None
        self.set_busy(False)

    def closeEvent(self, event):
        if self.worker is not None:
            self.statusBar().showMessage("请等待当前检索完成后关闭窗口。")
            event.ignore()
        else:
            self.local_session.stop()
            event.accept()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--release-smoke":
        from tools.release import smoke
        return smoke(Path(sys.argv[2]))
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.local_session.stop)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
