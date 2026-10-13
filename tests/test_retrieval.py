"""Test retrieval order and passage mapping without a model or GPU."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from langchain_core.documents import Document

from embed_query import build_context, retrieve, select_device, load_index, LocalModelError
from app_paths import application_root


class RetrievalTests(unittest.TestCase):
    def test_frozen_asset_root_uses_executable_directory(self):
        with patch("sys.frozen", True, create=True), \
             patch("sys.executable", str(Path("release/assistant.exe").resolve())):
            self.assertEqual(application_root(), Path("release").resolve())

    def setUp(self):
        self.model = Mock()
        self.model.encode.return_value = torch.tensor([[1.0, 0.0]])
        self.chunks = [
            Document(page_content="First.", metadata={"source": "a.txt", "start_index": 0}),
            Document(page_content="Second.", metadata={"source": "b.txt", "start_index": 5}),
        ]
        self.vectors = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        self.model._project_use_e5_prefixes = True

    def test_order_and_evidence_mapping(self):
        hits = retrieve("question", self.model, self.chunks, self.vectors)
        self.assertEqual([index for index, _, _ in hits], [1, 0])
        self.assertEqual([score for _, _, score in hits], [1.0, 0.0])
        self.model.encode.assert_called_once_with(
            ["query: question"], normalize_embeddings=True, convert_to_tensor=True
        )
        context = build_context([chunk for _, chunk, _ in hits])
        self.assertTrue(context.startswith("[1] Source: b.txt | Start index: 5\nSecond."))
        self.assertIn("[2] Source: a.txt | Start index: 0\nFirst.", context)

    def test_blank_question_does_not_encode(self):
        with self.assertRaises(ValueError):
            retrieve(" ", self.model, self.chunks, self.vectors)
        self.model.encode.assert_not_called()

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            retrieve("question", self.model, self.chunks, self.vectors, k=0)

    def test_mismatched_vectors(self):
        with self.assertRaises(ValueError):
            retrieve("question", self.model, self.chunks, self.vectors[:1])

    def test_auto_device_cpu_and_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device(), "cpu")
            with self.assertRaises(LocalModelError):
                select_device("cuda")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device(), "cuda")
            self.assertEqual(select_device("cpu"), "cpu")

    def test_missing_model_explained_without_download(self):
        with patch("pathlib.Path.is_file", return_value=False), \
             patch("embed_query.SentenceTransformer") as loader:
            with self.assertRaisesRegex(LocalModelError, "本地模型缺失"):
                load_index("bge-m3", [self.chunks[0]])
            loader.assert_not_called()

    def test_model_load_failure_is_sanitized(self):
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("embed_query.SentenceTransformer", side_effect=RuntimeError("private detail")):
            with self.assertRaises(LocalModelError) as raised:
                load_index("bge-m3", [self.chunks[0]], device="cpu")
        self.assertNotIn("private detail", str(raised.exception))
        self.assertIn("CPU", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
