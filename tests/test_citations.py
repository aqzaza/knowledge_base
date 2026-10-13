"""Local citation tests: no GPU inference or API requests."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from embed_query import answer_once, build_answer_prompt, check_citations


class CitationTests(unittest.TestCase):
    def test_answer_prompt_requires_part_by_part_evidence(self):
        prompt = build_answer_prompt("问题一和问题二？", "[1] Source: test.md | Start index: 0\n只有问题一的证据。")
        self.assertIn("Answer each part of the question separately.", prompt)
        self.assertIn("evidence is insufficient for that part", prompt)
        self.assertIn("Do not add explanations, causes, definitions, or conclusions from outside knowledge.", prompt)

    def test_valid(self):
        self.assertEqual(check_citations("Ingest [1]. Index [2].", 3), ([1, 2], []))

    def test_invalid(self):
        self.assertEqual(check_citations("Claims [0] [4].", 3), ([0, 4], [0, 4]))

    def test_repeated(self):
        self.assertEqual(check_citations("Claims [2] [1] [2].", 3), ([1, 2], []))

    def test_no_citations(self):
        self.assertEqual(check_citations("Insufficient evidence.", 3), ([], []))


class AnswerCitationTests(unittest.TestCase):
    def run_mock_answer(self, content):
        response = SimpleNamespace(
            content=content, response_metadata={"model_name": "mock"}, usage_metadata=None
        )
        output = StringIO()
        with (
            patch("dotenv.load_dotenv"),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-placeholder"}),
            patch("langchain_deepseek.ChatDeepSeek") as client,
            redirect_stdout(output),
        ):
            client.return_value.invoke.return_value = response
            result = answer_once("Local test prompt.", 3)
            self.assertEqual(result["answer"], content)
            self.assertEqual(
                (result["citations"], result["invalid_citations"]),
                check_citations(content, 3),
            )
            client.return_value.invoke.assert_called_once()
        return output.getvalue()

    def test_valid_answer(self):
        output = self.run_mock_answer("Ingest [1]. Index [2].")
        self.assertIn("Citation IDs valid: [1, 2]", output)
        self.assertIn("manual review", output)

    def test_invalid_answer(self):
        output = self.run_mock_answer("An unsupported claim [4].")
        self.assertIn("WARNING: Invalid citations: [4]", output)
        self.assertNotIn("Citation IDs valid:", output)

    def test_uncited_answer(self):
        output = self.run_mock_answer("Insufficient evidence.")
        self.assertIn("NOTICE: No citations", output)
        self.assertNotIn("Citation IDs valid:", output)


if __name__ == "__main__":
    unittest.main()
