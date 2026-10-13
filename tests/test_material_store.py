"""Persistence checks using temporary paths, no real user store or API."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
from langchain_core.documents import Document

from desktop_app import MainWindow
from material_store import load_materials, save_materials
from read_document import MAX_MATERIAL_FILES, MAX_MATERIAL_BYTES, MAX_IMPORT_BYTES


class MaterialStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.store = self.folder / "store" / "materials.json"

    def window(self):
        window = MainWindow(storage_path=self.store)
        self.addCleanup(window.close)
        return window

    def test_restart_remove_clear_and_no_key_saved(self):
        paths = []
        for i in range(2):
            path = self.folder / f"{i}.txt"
            path.write_text(f"中文测试资料 {i}", encoding="utf-8")
            paths.append(str(path))
        first = self.window()
        first.api_key_input.setText("private-test-placeholder")
        first.import_files(paths)
        self.assertNotIn("private-test-placeholder", self.store.read_text(encoding="utf-8"))
        first.close()
        # Restoration uses stored text, not the original file paths.
        Path(paths[0]).write_text("原文件已改变", encoding="utf-8")
        second = self.window()
        self.assertEqual(len(second.imported_documents), 2)
        self.assertIn("中文测试资料 0", second.import_preview.toPlainText())
        self.assertTrue(second.materials_dirty)
        self.assertIsNone(second.index)
        self.assertEqual(second.api_key_input.text(), "")
        second.remove_material(0)
        third = self.window()
        self.assertEqual([d.metadata["source"] for d in third.imported_documents], ["1.txt"])
        third.clear_materials()
        self.assertEqual(self.window().imported_documents, [])
        self.assertTrue(all(Path(p).exists() for p in paths))

    def test_write_failure_preserves_memory_and_saved_materials(self):
        path = self.folder / "one.txt"
        path.write_text("测试", encoding="utf-8")
        window = self.window()
        window.import_files([str(path)])
        original = self.store.read_bytes()
        with patch("desktop_app.save_materials", side_effect=OSError("private details")):
            window.remove_material(0)
            self.assertEqual(len(window.imported_documents), 1)
            self.assertIn("未生效", window.storage_status.text())
            self.assertNotIn("private details", window.storage_status.text())
        self.assertEqual(self.store.read_bytes(), original)

    def test_corrupt_store_preserved_and_writes_blocked(self):
        self.store.parent.mkdir()
        self.store.write_bytes(b"not json")
        window = self.window()
        self.assertTrue(window.storage_blocked)
        window.clear_materials()
        self.assertEqual(self.store.read_bytes(), b"not json")

    def test_invalid_metadata_duplicate_and_empty_rejected(self):
        for documents in [
            [Document(page_content="x", metadata={"source": "../a.txt", "bytes": 1})],
            [Document(page_content="", metadata={"source": "a.txt", "bytes": 1})],
            [Document(page_content="x", metadata={"source": name, "bytes": 1})
             for name in ["a.txt", "A.TXT"]],
        ]:
            with self.assertRaises(ValueError):
                save_materials(self.store, documents)
        self.assertFalse(self.store.exists())

    def test_bad_version_and_missing_store(self):
        self.assertEqual(load_materials(self.store), [])
        self.store.parent.mkdir()
        self.store.write_text(json.dumps({"version": 2, "materials": []}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_materials(self.store)

    def test_legacy_store_preserved_and_duplicate_legacy_import_rejected(self):
        doc = Document(page_content="旧版正文", metadata={"source": "old.txt", "bytes": 12})
        save_materials(self.store, [doc])
        original = self.store.read_bytes()
        window = self.window()
        self.assertFalse(window.storage_blocked)
        self.assertEqual(self.store.read_bytes(), original)
        path = self.folder / "old.txt"
        path.write_text(doc.page_content, encoding="utf-8")
        window.import_files([path])
        self.assertEqual(len(window.imported_documents), 1)
        self.assertIn("旧材料库", window.import_errors.toPlainText())
        self.assertEqual(self.store.read_bytes(), original)

    def test_capacity_and_raw_text_totals_are_bounded(self):
        docs = [Document(page_content="x", metadata={"source": f"{i}.txt", "bytes": 1})
                for i in range(MAX_MATERIAL_FILES)]
        save_materials(self.store, docs)
        self.assertEqual(len(load_materials(self.store)), MAX_MATERIAL_FILES)
        original = self.store.read_bytes()
        with self.assertRaises(ValueError):
            save_materials(self.store, docs + [Document(page_content="x", metadata={"source": "extra.txt", "bytes": 1})])
        too_large = [Document(page_content="x", metadata={"source": f"{i}.txt", "bytes": MAX_IMPORT_BYTES})
                     for i in range(MAX_MATERIAL_BYTES // MAX_IMPORT_BYTES + 1)]
        with self.assertRaises(ValueError):
            save_materials(self.store, too_large)
        with patch("material_store.MAX_MATERIAL_TEXT_BYTES", 1), self.assertRaises(ValueError):
            save_materials(self.store, docs[:2])
        self.assertEqual(self.store.read_bytes(), original)

    def test_import_identity_round_trip_and_validation(self):
        identity = "a" * 64
        doc = Document(page_content="x", metadata={"source": "one.txt", "bytes": 1, "import_id": identity})
        save_materials(self.store, [doc])
        self.assertEqual(load_materials(self.store)[0].metadata["import_id"], identity)
        self.assertNotIn(str(self.folder), self.store.read_text(encoding="utf-8"))
        for value in [None, "", "b" * 63, "G" * 64, 1]:
            invalid = Document(page_content="x", metadata={"source": "one.txt", "bytes": 1, "import_id": value})
            with self.subTest(value=value), self.assertRaises(ValueError):
                save_materials(self.store, [invalid])
        duplicate = Document(page_content="y", metadata={"source": "two.txt", "bytes": 1, "import_id": identity})
        with self.assertRaises(ValueError):
            save_materials(self.store, [doc, duplicate])


if __name__ == "__main__":
    unittest.main()
