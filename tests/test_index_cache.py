"""Test numeric cache and startup restore without loading a real model."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PySide6.QtWidgets import QApplication
from langchain_core.documents import Document

from index_cache import cached_model, load_vectors, save_vectors, cache_signature
from material_store import save_materials
from desktop_app import MainWindow


class IndexCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "index.npz"
        self.docs = [Document(page_content="项目代号是青竹", metadata={"source": "test.txt", "bytes": 24})]
        self.vectors = np.array([[1., 0.]], dtype=np.float32)

    def test_valid_cache_and_mismatch(self):
        save_vectors(self.path, "test-signature", "bge-m3", self.vectors)
        self.assertEqual(cached_model(self.path), "bge-m3")
        np.testing.assert_array_equal(load_vectors(self.path, "test-signature", 1, 2), self.vectors)
        self.assertIsNone(load_vectors(self.path, "changed", 1, 2))
        self.assertIsNone(load_vectors(self.path, "test-signature", 2, 2))
        self.assertIsNone(load_vectors(self.path, "test-signature", 1, 3))

    def test_corrupt_and_invalid_vectors(self):
        self.path.write_bytes(b"broken")
        self.assertIsNone(cached_model(self.path))
        self.assertIsNone(load_vectors(self.path, "test", 1, 2))
        for vector in [np.array([[np.nan, 0]]), np.array([[2., 0.]])]:
            save_vectors(self.path, "test", "bge-m3", vector)
            self.assertIsNone(load_vectors(self.path, "test", 1, 2))

    def test_content_and_model_change_signature(self):
        from split_documents import split_documents
        chunks = split_documents(self.docs)
        signature = cache_signature("bge-m3", chunks)
        self.assertNotEqual(signature, cache_signature("multilingual-e5-small", chunks))
        chunks[0].page_content += "变化"
        self.assertNotEqual(signature, cache_signature("bge-m3", chunks))

    def test_load_index_restores_without_document_encoding(self):
        from split_documents import split_documents
        from embed_query import load_index
        chunks = split_documents(self.docs)
        signature = cache_signature("bge-m3", chunks)
        save_vectors(self.path, signature, "bge-m3", self.vectors)
        model = Mock()
        model.device = torch.device("cpu")
        model.get_sentence_embedding_dimension.return_value = 2
        with patch("embed_query.SentenceTransformer", return_value=model):
            index = load_index("bge-m3", self.docs, cache_path=self.path, device="cpu")
        model.encode.assert_not_called()
        self.assertEqual(model._project_index_cache_status, "restored")
        self.assertEqual(len(index[1]), 1)

    def test_startup_automatically_requests_restore_and_clear_invalidates_cache(self):
        store = self.path.with_name("materials.json")
        save_materials(store, self.docs)
        save_vectors(self.path, "test", "multilingual-e5-small", self.vectors)
        with patch("desktop_app.IndexWorker") as worker:
            window = MainWindow(storage_path=store)
            self.addCleanup(window.close)
            self.assertEqual(window.model_selector.currentText(), "multilingual-e5-small")
            self.assertEqual(worker.call_args.kwargs["cache_path"], self.path)
            worker.return_value.start.assert_called_once()
            window.worker = None
            window.clear_materials()
            self.assertFalse(self.path.exists())

    def test_first_build_saves_cache_then_second_load_skips_encoding(self):
        from embed_query import load_index
        model = Mock()
        model.device = torch.device("cpu")
        model.get_sentence_embedding_dimension.return_value = 2
        model.encode.return_value = torch.from_numpy(self.vectors)
        with patch("embed_query.SentenceTransformer", return_value=model):
            load_index("bge-m3", self.docs, cache_path=self.path, device="cpu")
        model.encode.assert_called_once()
        self.assertTrue(self.path.exists())
        model.encode.reset_mock()
        with patch("embed_query.SentenceTransformer", return_value=model):
            load_index("bge-m3", self.docs, cache_path=self.path, device="cpu")
        model.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
