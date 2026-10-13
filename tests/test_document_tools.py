"""Exercise the real search wrapper with synthetic CPU vectors, no API/GPU."""

import unittest
from unittest.mock import Mock

import torch
from langchain_core.documents import Document
from pydantic import ValidationError

from document_tools import make_document_tools, make_search_documents_tool


class SearchToolTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.encode.return_value = torch.tensor([[1.0, 0.0]])
        self.chunks = [
            Document(page_content="预算为300。", metadata={"source": "budget.xlsx", "start_index": 12}),
            Document(page_content="项目代号是青竹。", metadata={"source": "project.docx", "start_index": 0}),
        ]
        self.vectors = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        self.search = make_search_documents_tool((self.model, self.chunks, self.vectors, False))

    def test_chinese_question_and_original_mapping(self):
        result = self.search.invoke({"question": "项目代号是什么？"})
        first = result["results"][0]
        self.assertEqual(self.search.name, "search_documents")
        self.assertEqual(first["evidence_id"], "chunk-2")
        self.assertEqual(first["source"], "project.docx")
        self.assertEqual(first["start_index"], 0)
        self.assertEqual(first["excerpt"], "项目代号是青竹。")
        self.assertEqual(first["score"], 1.0)
        self.model.encode.assert_called_once_with(
            ["项目代号是什么？"], normalize_embeddings=True, convert_to_tensor=True
        )

    def test_ids_stable_when_order_changes(self):
        first = self.search.invoke({"question": "项目代号？", "top_k": 1})
        self.model.encode.return_value = torch.tensor([[0.0, 1.0]])
        second = self.search.invoke({"question": "预算？", "top_k": 2})
        self.assertEqual(first["results"][0]["evidence_id"], second["results"][1]["evidence_id"])
        self.assertEqual(second["results"][0]["evidence_id"], "chunk-1")

    def test_validation_before_encoding(self):
        for values in [{"question": " "}, {"question": "x" * 1001},
                       {"question": "问题", "top_k": 0},
                       {"question": "问题", "top_k": 6},
                       {"question": "问题", "top_k": True}]:
            with self.subTest(values=values), self.assertRaises((ValueError, ValidationError)):
                self.search.invoke(values)
        self.model.encode.assert_not_called()

    def test_excerpt_bounded_and_index_snapshot(self):
        self.chunks[1].page_content = "原文" * 200
        search = make_search_documents_tool((self.model, self.chunks, self.vectors, False))
        self.chunks[1].page_content = "之后修改"
        result = search.invoke({"question": "问题", "top_k": 1})["results"][0]
        self.assertEqual(result["excerpt"], "原文" * 100)
        self.assertTrue(result["truncated"])

    def test_empty_index_rejected(self):
        with self.assertRaisesRegex(ValueError, "No indexed passages"):
            make_search_documents_tool((self.model, [], self.vectors[:0], False))

    def test_read_passage_requires_search_id_and_returns_full_text(self):
        search, read = make_document_tools((self.model, self.chunks, self.vectors, False))
        result = search.invoke({"question": "项目代号是什么？", "top_k": 1})["results"][0]
        passage = read.invoke({"evidence_id": result["evidence_id"]})
        self.assertEqual(passage["text"], "项目代号是青竹。")
        self.assertEqual(passage["source"], "project.docx")
        for invalid in ["chunk-0", "chunk-99", "budget.xlsx", "../chunk-1"]:
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, ValidationError)):
                read.invoke({"evidence_id": invalid})

    def test_agent_rejects_unsafe_tool_limit_before_api_call(self):
        from document_tools import run_agent
        with self.assertRaisesRegex(ValueError, "between 1 and 3"):
            run_agent("问题", (self.model, self.chunks, self.vectors, False), max_tool_calls=4)

    def test_api_configuration_never_uses_environment_key(self):
        from unittest.mock import patch
        from document_tools import validate_api_config, run_agent
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "owner-placeholder"}):
            with self.assertRaisesRegex(ValueError, "自己的 API Key"):
                run_agent("问题", (self.model, self.chunks, self.vectors, False))
            self.assertEqual(validate_api_config({"api_key": " user-placeholder "})["api_key"],
                             "user-placeholder")
        for base in ["http://example.com", "https://user:pass@example.com", "https://example.com?key=x"]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                validate_api_config({"api_key": "placeholder", "base_url": base})


if __name__ == "__main__":
    unittest.main()
