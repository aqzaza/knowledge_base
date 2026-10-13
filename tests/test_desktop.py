"""Offline desktop tests with mocked model loading and retrieval."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from PySide6.QtWidgets import QApplication
from langchain_core.documents import Document

from desktop_app import MainWindow, RetrievalWorker, IndexWorker
from read_document import MAX_MATERIAL_FILES


class DesktopTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        replacement = patch("desktop_app.default_material_path", return_value=Path(folder.name) / "materials.json")
        replacement.start()
        self.addCleanup(replacement.stop)

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_empty_question_does_not_start_worker(self):
        window = MainWindow()
        window.on_query()
        self.assertIsNone(window.worker)
        self.assertEqual(window.output.toPlainText(), "请输入问题。")
        window.close()

    def test_import_preview_and_errors_do_not_replace_index(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            good = folder / "中文.md"
            good.write_text("# 中文资料\n测试正文", encoding="utf-8-sig")
            empty = folder / "empty.txt"
            empty.write_text("  ", encoding="utf-8")
            invalid = folder / "invalid.txt"
            invalid.write_bytes(b"\xff\xfe")
            unsupported = folder / "code.py"
            unsupported.write_text("print(1)", encoding="utf-8")
            large = folder / "large.txt"
            large.write_bytes(b"a" * (1024 * 1024 + 1))
            window = MainWindow()
            window.index = index = ("existing index",)
            with patch("embed_query.load_index") as load, patch("embed_query.answer_once") as answer:
                window.import_files([str(p) for p in [good, empty, invalid, unsupported, large]])
                load.assert_not_called()
                answer.assert_not_called()
            self.assertIs(window.index, index)
            self.assertEqual(len(window.imported_documents), 1)
            self.assertEqual(window.imported_documents[0].metadata["source"], good.name)
            self.assertEqual(window.import_preview.toPlainText(), "# 中文资料\n测试正文")
            self.assertIn("UTF-8", window.import_errors.toPlainText())
            self.assertIn("正文为空", window.import_errors.toPlainText())
            self.assertIn("1 MiB", window.import_errors.toPlainText())
            window.import_files([str(good)] * (MAX_MATERIAL_FILES + 1))
            self.assertEqual(len(window.imported_documents), 1)
            self.assertIn("最多", window.import_errors.toPlainText())
            window.close()

    def test_import_total_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(6):
                path = Path(directory) / f"{i}.txt"
                path.write_bytes(b"a" * (1024 * 1024))
                paths.append(str(path))
            window = MainWindow()
            with patch("desktop_app.MAX_MATERIAL_BYTES", 2 * 1024 * 1024):
                window.import_files(paths)
            self.assertEqual(len(window.imported_documents), 2)
            self.assertIn("2 MiB", window.import_errors.toPlainText())
            window.close()

    def test_empty_start_and_incremental_material_management(self):
        window = MainWindow()
        self.assertEqual(window.active_documents, [])
        self.assertFalse(window.query_button.isEnabled())
        window.question_input.setPlainText("问题")
        with patch("desktop_app.RetrievalWorker") as worker:
            window.on_query()
            worker.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(3):
                path = Path(directory) / f"{i}.txt"
                path.write_text(f"项目 {i} 的负责人是测试人员 {i}。", encoding="utf-8")
                paths.append(str(path))
                window.import_files([str(path)])
                self.assertEqual(len(window.imported_documents), i + 1)
            docs = list(window.imported_documents)
            window.on_index_ready(("model", docs, "vectors", False), docs)
            self.assertTrue(window.query_button.isEnabled())
            window.import_files([paths[0]])
            self.assertEqual(len(window.imported_documents), 3)
            self.assertFalse(window.materials_dirty)
            self.assertIn("已导入", window.import_errors.toPlainText())
            window.remove_material(1)
            self.assertEqual([d.metadata["source"] for d in window.imported_documents], ["0.txt", "2.txt"])
            self.assertTrue(Path(paths[1]).exists())
            self.assertFalse(window.query_button.isEnabled())
            self.assertEqual(window.hits, [])
            window.on_index_ready(("model", list(window.imported_documents), "vectors", False), list(window.imported_documents))
            self.assertTrue(window.query_button.isEnabled())
            window.clear_materials()
            self.assertEqual(window.imported_documents, [])
            self.assertEqual(window.active_documents, [])
            self.assertIsNone(window.index)
            self.assertFalse(window.query_button.isEnabled())
        window.close()

    def test_many_same_basename_files_import_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = []
            for i in range(140):
                subfolder = folder / f"chapter_{i}"
                subfolder.mkdir()
                path = subfolder / "index.md"
                path.write_text(f"章节 {i} 的测试资料。", encoding="utf-8")
                paths.append(path)
            window = MainWindow(storage_path=folder / "materials.json")
            self.addCleanup(window.close)
            window.import_files(paths)
            self.assertEqual(len(window.imported_documents), 140)
            names = [d.metadata["source"] for d in window.imported_documents]
            self.assertEqual(len({name.casefold() for name in names}), 140)
            self.assertEqual(names[0], "index.md")
            self.assertIn("chapter_1", names[1])
            self.assertTrue(all(p.name == "index.md" for p in paths))
            window.close()
            restored = MainWindow(storage_path=folder / "materials.json")
            self.addCleanup(restored.close)
            self.assertEqual([d.metadata["source"] for d in restored.imported_documents], names)
            restored.import_files(paths)
            self.assertEqual(len(restored.imported_documents), 140)
            self.assertIn("已导入", restored.import_errors.toPlainText())

    def test_same_named_identical_files_from_different_folders_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = []
            for name in ["a", "b"]:
                subfolder = folder / name
                subfolder.mkdir()
                path = subfolder / "same.txt"
                path.write_text("相同的正文", encoding="utf-8")
                paths.append(path)
            window = MainWindow(storage_path=folder / "materials.json")
            self.addCleanup(window.close)
            window.import_files(paths)
            self.assertEqual(len(window.imported_documents), 2)
            self.assertNotEqual(window.imported_documents[0].metadata["source"],
                                window.imported_documents[1].metadata["source"])
            self.assertNotEqual(window.imported_documents[0].metadata["import_id"],
                                window.imported_documents[1].metadata["import_id"])

    def test_folder_import_is_recursive_and_busy_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "nested").mkdir()
            (folder / "a.txt").write_text("根目录资料", encoding="utf-8")
            (folder / "nested" / "a.txt").write_text("子目录资料", encoding="utf-8")
            (folder / "nested" / "code.py").write_text("raise RuntimeError()", encoding="utf-8")
            window = MainWindow()
            self.addCleanup(window.close)
            window.import_folder(folder)
            self.assertEqual(len(window.imported_documents), 2)
            window.worker = object()
            try:
                with patch("read_document.collect_import_paths") as collect:
                    window.import_folder(folder)
                    collect.assert_not_called()
                window.set_busy(True)
                self.assertFalse(window.import_folder_button.isEnabled())
            finally:
                window.worker = None

    def test_collection_count_and_text_limits_preserve_valid_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            paths = []
            for i in range(3):
                path = folder / f"{i}.txt"
                path.write_text("中文", encoding="utf-8")
                paths.append(path)
            window = MainWindow()
            self.addCleanup(window.close)
            with patch("desktop_app.MAX_MATERIAL_TEXT_BYTES", 6):
                window.import_files(paths[:2])
            self.assertEqual(len(window.imported_documents), 1)
            self.assertIn("正文总量", window.import_errors.toPlainText())
            with patch("desktop_app.MAX_MATERIAL_FILES", 1):
                window.import_files(paths[2:])
            self.assertEqual(len(window.imported_documents), 1)
            self.assertIn("最多容纳 1", window.import_errors.toPlainText())

    def test_many_material_header_stays_compact(self):
        window = MainWindow()
        self.addCleanup(window.close)
        documents = [Document(page_content="资料", metadata={"source": f"{i}.md"}) for i in range(140)]
        window.on_index_ready(("model", documents, "vectors", False), documents)
        self.assertIn("140", window.active_source.text())
        self.assertLess(len(window.active_source.text()), 100)
        self.assertNotIn("139.md", window.active_source.text())

    def test_import_build_query_answer_end_to_end(self):
        window = MainWindow()
        path = Path(__file__).resolve().parents[1] / "data" / "import_qa_test.txt"
        window.import_files([str(path)])
        documents = window.imported_documents
        chunk = Document(page_content=documents[0].page_content,
                         metadata={"source": path.name, "start_index": 0})
        index = ("model", [chunk], "vectors", False)
        result = {"answer": "本测试项目的代号是青竹 [1]。", "model": "mock",
                  "usage": {"total_tokens": 20}, "citations": [1], "invalid_citations": [],
                  "trace": [{"tool": "search_documents"}], "hits": [(0, chunk, 0.9)]}
        with patch("embed_query.load_index", return_value=index) as load:
            worker = IndexWorker("bge-m3", documents)
            worker.succeeded.connect(window.on_index_ready)
            worker.run()
            load.assert_called_once_with("bge-m3", documents=documents)
        self.assertEqual(window.active_documents, documents)
        self.assertIn(path.name, window.active_source.text())
        with patch("document_tools.run_agent", return_value=result) as agent:
            worker = RetrievalWorker("bge-m3", "本测试项目的代号是什么？", window.index,
                                     generate=True, documents=window.active_documents)
            worker.succeeded.connect(window.on_results)
            worker.answered.connect(window.on_answer)
            worker.run()
            agent.assert_called_once()
        self.assertIn("青竹", window.answer_view.toPlainText())
        self.assertIn("[1]", window.answer_view.toPlainText())
        self.assertEqual(window.hits[0][1].metadata["source"], path.name)
        window.model_selector.setCurrentText("multilingual-e5-small")
        self.assertIsNone(window.index)
        self.assertEqual(window.active_documents, documents)
        with patch("embed_query.load_index", return_value=index) as load, patch("embed_query.retrieve", return_value=[]):
            worker = RetrievalWorker("multilingual-e5-small", "问题", None, documents=documents)
            worker.run()
            load.assert_called_once_with("multilingual-e5-small", documents=documents)
        window.close()

    def test_failed_rebuild_preserves_active_corpus(self):
        window = MainWindow()
        window.index = old_index = ("old",)
        window.active_documents = old_documents = ["old document"]
        with patch("embed_query.load_index", side_effect=RuntimeError("secret")):
            worker = IndexWorker("bge-m3", ["new document"])
            worker.failed.connect(window.on_index_failure)
            worker.run()
        self.assertIs(window.index, old_index)
        self.assertIs(window.active_documents, old_documents)
        self.assertIn("原知识库保留", window.import_errors.toPlainText())
        self.assertNotIn("secret", window.import_errors.toPlainText())
        window.close()

    def test_model_switch_clears_index_and_evidence(self):
        window = MainWindow()
        chunk = Document(page_content="中文证据", metadata={"source": "test.md", "start_index": 12})
        window.on_results(("model",), [(9, chunk, 0.5)])
        self.assertIn("中文证据", window.output.toPlainText())
        self.assertIn("字符起点 12", window.evidence_metadata.text())
        window.model_selector.setCurrentText("multilingual-e5-base")
        self.assertIsNone(window.index)
        self.assertEqual(window.output.toPlainText(), "")
        window.close()

    def test_evidence_selection_preserves_original_text(self):
        window = MainWindow()
        chunks = [Document(page_content=text, metadata={"source": "test.md", "start_index": i})
                  for i, text in enumerate(["**原文**", "<script>不执行</script>"])]
        window.on_results(("model",), [(i, c, 0.5) for i, c in enumerate(chunks)])
        self.assertEqual(window.output.toPlainText(), chunks[0].page_content)
        window.evidence_list.setCurrentRow(1)
        self.assertEqual(window.output.toPlainText(), chunks[1].page_content)
        window.close()

    def test_worker_loads_selected_model_and_reuses_index(self):
        index = ("model", [Document(page_content="证据", metadata={"source": "test", "start_index": 0})],
                 torch.tensor([[1.0]]), False)
        with patch("embed_query.load_index", return_value=index) as load, patch("embed_query.retrieve", return_value=[]) as search:
            worker = RetrievalWorker("bge-m3", "中文问题", None)
            worker.run()
            load.assert_called_once_with("bge-m3")
            search.assert_called_once_with("中文问题", *index, k=3)
            load.reset_mock()
            worker.run()
            load.assert_not_called()

    def test_worker_reports_failure(self):
        with patch("embed_query.load_index", side_effect=RuntimeError("private details")):
            worker = RetrievalWorker("bge-m3", "问题", None)
            messages = []
            worker.failed.connect(messages.append)
            worker.run()
            self.assertIn("RuntimeError", messages[0])
            self.assertNotIn("private details", messages[0])

    def test_markdown_rendering_keeps_raw_evidence(self):
        window = MainWindow()
        text = "- **Knowledge base:** preprocess documents\n\n```python\nprint('test')\n```"
        chunk = Document(page_content=text, metadata={"source": "test.md", "start_index": 0})
        window.on_results(None, [(0, chunk, 0.5)])
        self.assertEqual(window.output.toPlainText(), text)
        self.assertNotIn("**", window.preview.toPlainText())
        self.assertIn("print('test')", window.preview.toPlainText())
        self.assertIsNone(window.preview.document().loadResource(2, None))
        self.assertEqual(window.evidence_tabs.currentIndex(), 0)
        window.close()

    def test_math_rendering_preserves_raw_and_code(self):
        window = MainWindow()
        text = r"梯度范数不超过$\theta$，方向为$\mathbf{g}$。" + "\n\n" + r"`$\theta$`"
        chunk = Document(page_content=text, metadata={"source": "math.md", "start_index": 0})
        window.on_results(None, [(0, chunk, 0.5)])
        self.assertEqual(window.output.toPlainText(), text)
        rendered = window.preview.toPlainText()
        self.assertEqual(rendered.count("\ufffc"), 2)
        self.assertEqual(rendered.count(r"$\theta$"), 1)
        self.assertNotIn("formula", rendered)
        self.assertIsNone(window.preview.document().loadResource(2, None))
        window.close()

    def test_unsupported_math_and_fenced_code_are_preserved(self):
        window = MainWindow()
        text = r"$\unsupported{x}$" + "\n\n```latex\n" + r"$\theta$" + "\n```"
        chunk = Document(page_content=text, metadata={"source": "math.md", "start_index": 0})
        window.on_results(None, [(0, chunk, 0.5)])
        self.assertIn(r"$\unsupported{x}$", window.preview.toPlainText())
        self.assertIn(r"$\theta$", window.preview.toPlainText())
        self.assertEqual(window.output.toPlainText(), text)
        window.close()

    def test_default_worker_never_calls_answer(self):
        with patch("embed_query.retrieve", return_value=[]), patch("embed_query.answer_once") as answer:
            RetrievalWorker("bge-m3", "问题", ("model", [], "vectors", False)).run()
            answer.assert_not_called()
        window = MainWindow()
        self.assertFalse(window.generate_checkbox.isChecked())
        window.close()

    def test_paid_worker_calls_once_with_numbered_evidence(self):
        chunk = Document(page_content="证据", metadata={"source": "test.md", "start_index": 0})
        result = {"answer": "答案 [1]", "model": "mock", "usage": None,
                  "citations": [1], "invalid_citations": [], "trace": [],
                  "hits": [(0, chunk, 0.5)]}
        with patch("document_tools.run_agent", return_value=result) as agent:
            worker = RetrievalWorker("bge-m3", "中文问题", ("model", [chunk], "vectors", False), generate=True)
            received = []
            worker.answered.connect(received.append)
            worker.run()
            agent.assert_called_once()
            self.assertEqual(received, [result])

    def test_paid_worker_uses_agent_result_and_trace(self):
        chunk = Document(page_content="证据", metadata={"source": "test.md", "start_index": 0})
        result = {"answer": "答案 [1]", "model": "mock", "usage": {"total_tokens": 2},
                  "citations": [1], "invalid_citations": [], "trace": [
                      {"tool": "search_documents", "args": {"question": "问题"},
                       "result": {"keys": ["results"], "count": 1}}
                  ], "hits": [(0, chunk, 0.5)]}
        with patch("document_tools.run_agent", return_value=result) as agent:
            worker = RetrievalWorker("bge-m3", "问题", ("model", [chunk], "vectors", False), generate=True)
            received = []
            worker.answered.connect(received.append)
            worker.run()
            agent.assert_called_once_with("问题", ("model", [chunk], "vectors", False),
                                          max_tool_calls=3, api_config=None)
            self.assertEqual(received, [result])

    def test_generation_failure_keeps_evidence(self):
        window = MainWindow()
        chunk = Document(page_content="证据", metadata={"source": "test.md", "start_index": 0})
        index = ("model", [chunk], "vectors", False)
        with patch("document_tools.run_agent", side_effect=ValueError("secret")):
            worker = RetrievalWorker("bge-m3", "问题", index, generate=True)
            worker.failed.connect(window.on_failure)
            worker.run()
        self.assertIn("检索失败", window.output.toPlainText())
        self.assertNotIn("secret", window.output.toPlainText())
        window.close()

    def test_answer_display_and_invalid_citations(self):
        window = MainWindow()
        window.on_answer({"answer": "**答案** [4]", "model": "mock", "usage": {"total_tokens": 10},
                          "citations": [4], "invalid_citations": [4]})
        self.assertIn("无效引用", window.citation_status.text())
        self.assertIn("total_tokens", window.call_details.toPlainText())
        self.assertEqual(window.result_tabs.currentIndex(), 0)
        window.on_model_changed("bge-m3")
        self.assertEqual(window.answer_view.toPlainText(), "")
        window.close()

    def test_api_settings_block_missing_key_and_pass_explicit_config(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.index = ("test index",)
        window.materials_dirty = False
        window.active_documents = ["test document"]
        window.question_input.setPlainText("问题")
        window.generate_checkbox.setChecked(True)
        with patch("desktop_app.RetrievalWorker") as worker:
            window.on_query()
            worker.assert_not_called()
            self.assertEqual(window.result_tabs.currentIndex(), window.api_tab_index)
            window.api_key_input.setText("session-test-key")
            window.on_query()
            self.assertEqual(worker.call_args.kwargs["api_config"], {
                "api_key": "session-test-key", "model": "deepseek-flash",
                "base_url": "https://api.deepseek.com"})
            window.worker = None
        window.set_busy(False)
        window.clear_api_button.click()
        self.assertEqual(window.api_key_input.text(), "")
        self.assertNotIn("session-test-key", window.api_status.text())
        second = MainWindow()
        self.assertEqual(second.api_key_input.text(), "")
        second.close()


if __name__ == "__main__":
    unittest.main()
