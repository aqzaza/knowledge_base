"""Offline local-answer and desktop routing tests with isolated user data."""

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication
from langchain_core.documents import Document

from desktop_app import MainWindow, RetrievalWorker
from local_answer import answer_local, LocalAnswerError, LocalAnswerSession


class LocalAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.chunk = Document(page_content="代号是青竹。", metadata={"source": "test.txt", "start_index": 0})
        self.processes = []
        self.requests = []
        self.log = "load_tensors: offloaded 37/37 layers to GPU\n"
        self.session = LocalAnswerSession()
        self.addCleanup(self.session.stop)

    def start_fake(self, command, **kwargs):
        process = Mock(pid=100 + len(self.processes), stdout=io.StringIO(self.log))
        process.poll.return_value = None
        process.wait.side_effect = lambda **_: setattr(process.poll, "return_value", 0)
        self.processes.append(process)
        return process

    def fake_request(self, path, payload=None, timeout=180):
        self.requests.append((path, payload, timeout))
        if path == "/health":
            self.session._reader.join(timeout=1)
            return {"status": "ok"}
        if path == "/props":
            return {}
        return {"choices": [{"message": {"content": "青竹 [1]。"}}]}

    def fake_engine(self):
        return patch("local_answer.subprocess.Popen", side_effect=self.start_fake)

    def test_offline_command_prompt_and_citations(self):
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine() as start, \
             patch.object(self.session, "_request", side_effect=self.fake_request):
            result = answer_local("代号？", [self.chunk], device="cpu", session=self.session)
        command = start.call_args.args[0]
        self.assertIn("llama-server.exe", command[0])
        self.assertIn("--offline", command)
        self.assertIn("--no-agent", command)
        self.assertIn("--no-ui", command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--parallel") + 1], "1")
        self.assertNotIn(self.session._token, command)
        self.assertEqual(start.call_args.kwargs["env"]["LLAMA_API_KEY"], self.session._token)
        path, payload, timeout = self.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(timeout, 180)
        self.assertIn("[1] Source: test.txt", payload["messages"][1]["content"])
        self.assertIn("代号是青竹", payload["messages"][1]["content"])
        self.assertIn("证据没有明确支持的部分必须明确说", payload["messages"][0]["content"])
        self.assertIn("不得补充原因、定义、解释或推断", payload["messages"][0]["content"])
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(result["citations"], [1])
        self.assertEqual(result["mode"], "local")
        self.assertEqual(result["device"], "CPU")
        self.assertEqual(result["gpu_layers"], 0)

    def test_cuda_command_and_actual_offload(self):
        with patch("pathlib.Path.is_file", return_value=True), \
             self.fake_engine() as start, patch.object(self.session, "_request", side_effect=self.fake_request):
            result = self.session.answer("代号？", [self.chunk], device="cuda")
        command = start.call_args.args[0]
        self.assertEqual(command[command.index("-ngl") + 1], "99")
        self.assertIn("llama-cuda13", command[0])
        self.assertEqual(result["device"], "CUDA")
        self.assertEqual(result["gpu_layers"], 37)

    def test_auto_prefers_cuda_and_reports_partial_offload(self):
        self.log = "load_tensors: offloaded 20/37 layers to GPU"
        with patch("pathlib.Path.is_file", return_value=True), \
             self.fake_engine() as start, patch.object(self.session, "_request", side_effect=self.fake_request):
            result = self.session.answer("代号？", [self.chunk])
        self.assertIn("llama-cuda13", start.call_args.args[0][0])
        self.assertEqual(result["gpu_layers"], 20)

    def test_cuda_without_offload_does_not_claim_gpu(self):
        for stderr in ["", "load_tensors: offloaded 0/37 layers to GPU"]:
            self.log = stderr
            with self.subTest(stderr=stderr), patch("pathlib.Path.is_file", return_value=True), \
                 self.fake_engine() as start, patch.object(self.session, "_request", side_effect=self.fake_request):
                with self.assertRaisesRegex(LocalAnswerError, "未确认"):
                    self.session.answer("代号？", [self.chunk], device="cuda")
                start.assert_called_once()
                self.assertIsNone(self.session.pid)

    def test_cuda_failure_no_automatic_retry(self):
        with patch("pathlib.Path.is_file", return_value=True), \
             self.fake_engine() as start, patch.object(self.session, "_request", side_effect=ConnectionError), \
             patch("local_answer.time.sleep", side_effect=lambda _: setattr(self.processes[-1].poll, "return_value", 1)):
            with self.assertRaisesRegex(LocalAnswerError, "CUDA 推理失败"):
                self.session.answer("代号？", [self.chunk], device="cuda")
            start.assert_called_once()
            self.assertIsNone(self.session.pid)

    def test_auto_uses_cpu_when_cuda_engine_absent(self):
        with patch("pathlib.Path.is_file", side_effect=lambda path: "llama-cuda13" not in str(path),
                   autospec=True), self.fake_engine() as start, \
             patch.object(self.session, "_request", side_effect=self.fake_request):
            result = self.session.answer("代号？", [self.chunk])
        command = start.call_args.args[0]
        self.assertIn("llama-bin", command[0])
        self.assertEqual(command[command.index("-ngl") + 1], "0")
        self.assertEqual(result["device"], "CPU")

    def test_missing_engine_and_timeout(self):
        with patch("pathlib.Path.is_file", return_value=False), self.fake_engine() as start, self.assertRaises(LocalAnswerError):
            self.session.answer("问题", [self.chunk])
        start.assert_not_called()
        def request(path, payload=None, timeout=180):
            if payload is not None:
                raise TimeoutError()
            return self.fake_request(path, payload, timeout)
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine(), \
             patch.object(self.session, "_request", side_effect=request):
            with self.assertRaisesRegex(LocalAnswerError, "180 秒"):
                self.session.answer("问题", [self.chunk])
        self.assertIsNone(self.session.pid)
        self.processes[-1].terminate.assert_called_once()

    def test_two_questions_reuse_process_without_conversation_history(self):
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine() as start, \
             patch.object(self.session, "_request", side_effect=self.fake_request):
            self.session.answer("第一次问题", [self.chunk], "cuda")
            pid = self.session.pid
            second = Document(page_content="预算是300。", metadata=self.chunk.metadata)
            self.session.answer("第二次问题", [second], "cuda")
        self.assertEqual(self.session.pid, pid)
        start.assert_called_once()
        prompt = self.requests[-1][1]["messages"]
        self.assertEqual(len(prompt), 2)
        self.assertNotIn("第一次问题", prompt[1]["content"])
        self.assertNotIn("青竹", prompt[1]["content"])
        self.assertIn("预算是300", prompt[1]["content"])

    def test_device_change_stops_old_process_and_starts_new(self):
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine() as start, \
             patch.object(self.session, "_request", side_effect=self.fake_request):
            self.session.answer("问题", [self.chunk], "cuda")
            first = self.processes[0]
            self.session.answer("问题", [self.chunk], "cpu")
        first.terminate.assert_called_once()
        self.assertEqual(start.call_count, 2)
        self.assertEqual(self.session.pid, 101)
        self.assertEqual(self.session._device, "CPU")

    def test_ready_session_ignores_later_offload_like_log_text(self):
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine(), \
             patch.object(self.session, "_request", side_effect=self.fake_request):
            self.session.answer("问题", [self.chunk], "cuda")
        self.session._read_log(io.StringIO("offloaded 99/99 layers to GPU\n"))
        self.assertEqual(self.session._gpu_layers, 37)

    def test_stop_is_idempotent_and_kills_when_termination_times_out(self):
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine(), \
             patch.object(self.session, "_request", side_effect=self.fake_request):
            self.session.answer("问题", [self.chunk], "cpu")
        process = self.processes[-1]
        process.wait.side_effect = [subprocess.TimeoutExpired("test", 3), 0]
        self.session.stop()
        self.session.stop()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertTrue(process.stdout.closed)
        self.assertIsNone(self.session.pid)
        self.assertIsNone(self.session._token)

    def test_startup_timeout_stops_process(self):
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine() as start, \
             patch("local_answer.time.monotonic", side_effect=[0, 181]):
            with self.assertRaisesRegex(LocalAnswerError, "加载超过"):
                self.session.answer("问题", [self.chunk], "cpu")
        start.assert_called_once()
        self.processes[-1].terminate.assert_called_once()

    def test_startup_requires_authenticated_service(self):
        def request(path, payload=None, timeout=180):
            if path == "/props":
                raise LocalAnswerError("unauthorized")
            return self.fake_request(path, payload, timeout)
        with patch("pathlib.Path.is_file", return_value=True), self.fake_engine(), \
             patch.object(self.session, "_request", side_effect=request), \
             patch("local_answer.time.monotonic", side_effect=[0, 1, 181]), \
             patch("local_answer.time.sleep"):
            with self.assertRaisesRegex(LocalAnswerError, "加载超过"):
                self.session.answer("问题", [self.chunk], "cpu")
        self.assertFalse(any(path == "/v1/chat/completions" for path, _, _ in self.requests))
        self.assertIsNone(self.session.pid)

    def test_bad_response_stops_process(self):
        for response in [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}]:
            def request(path, payload=None, timeout=180):
                return response if payload is not None else self.fake_request(path, payload, timeout)
            with self.subTest(response=response), patch("pathlib.Path.is_file", return_value=True), \
                 self.fake_engine(), patch.object(self.session, "_request", side_effect=request):
                with self.assertRaises(LocalAnswerError):
                    self.session.answer("问题", [self.chunk], "cpu")
                self.assertIsNone(self.session.pid)

    def test_validation_does_not_start_process(self):
        with self.fake_engine() as start:
            for question, chunks in [("", [self.chunk]), ("x" * 1001, [self.chunk]), ("问题", [])]:
                with self.assertRaises(LocalAnswerError):
                    self.session.answer(question, chunks)
        start.assert_not_called()

    def test_single_call_facade_always_stops_owned_session(self):
        with patch("local_answer.LocalAnswerSession") as session:
            answer_local("问题", [self.chunk])
            session.return_value.stop.assert_called_once()
            session.return_value.answer.side_effect = LocalAnswerError("failed")
            with self.assertRaises(LocalAnswerError):
                answer_local("问题", [self.chunk])
            self.assertEqual(session.return_value.stop.call_count, 2)

    def test_http_uses_loopback_token_and_refuses_redirect(self):
        self.session._port = 12345
        self.session._token = "session-test"
        response = Mock(status=200)
        response.read.return_value = json.dumps({"status": "ok"}).encode()
        with patch("local_answer.http.client.HTTPConnection") as connection:
            connection.return_value.getresponse.return_value = response
            self.assertEqual(self.session._request("/health", timeout=2), {"status": "ok"})
            connection.assert_called_once_with("127.0.0.1", 12345, timeout=2)
            headers = connection.return_value.request.call_args.kwargs["headers"]
            self.assertEqual(headers["Authorization"], "Bearer session-test")
            response.status = 302
            with self.assertRaises(LocalAnswerError):
                self.session._request("/health")
            self.assertEqual(connection.return_value.close.call_count, 2)

    def test_local_worker_does_not_use_agent(self):
        index = ("model", [self.chunk], "vectors", False)
        answer = {"answer": "青竹 [1]", "mode": "local"}
        with patch("embed_query.retrieve", return_value=[(0, self.chunk, .5)]), \
             patch("local_answer.answer_local", return_value=answer) as local, \
             patch("document_tools.run_agent") as agent:
            worker = RetrievalWorker("bge-m3", "代号？", index, local=True)
            received = []
            worker.answered.connect(received.append)
            worker.run()
        agent.assert_not_called()
        local.assert_called_once_with("代号？", [self.chunk], device="auto", session=None)
        self.assertEqual(received, [answer])

    def test_local_failure_keeps_evidence(self):
        worker = RetrievalWorker("bge-m3", "代号？", ("model", [self.chunk], "vectors", False), local=True)
        hits, errors = [], []
        worker.succeeded.connect(lambda index, found: hits.extend(found))
        worker.answer_failed.connect(errors.append)
        with patch("embed_query.retrieve", return_value=[(0, self.chunk, .5)]), \
             patch("local_answer.answer_local", side_effect=LocalAnswerError("模型缺失")):
            worker.run()
        self.assertEqual(len(hits), 1)
        self.assertEqual(errors, ["模型缺失"])

    def test_ui_local_mode_does_not_require_key(self):
        with tempfile.TemporaryDirectory() as folder:
            window = MainWindow(storage_path=Path(folder) / "materials.json")
            window.index = ("model", [self.chunk], "vectors", False)
            window.active_documents = [self.chunk]
            window.materials_dirty = False
            window.question_input.setPlainText("代号？")
            window.answer_mode.setCurrentIndex(1)
            with patch("desktop_app.RetrievalWorker") as worker:
                window.on_query()
                self.assertTrue(worker.call_args.kwargs["local"])
                self.assertFalse(worker.call_args.kwargs["generate"])
                self.assertIsNone(worker.call_args.kwargs["api_config"])
                self.assertEqual(worker.call_args.kwargs["local_device"], "auto")
                self.assertIs(worker.call_args.kwargs["local_session"], window.local_session)
                window.worker = None
            window.set_busy(False)
            self.assertTrue(window.local_device_selector.isEnabled())
            window.answer_mode.setCurrentIndex(0)
            self.assertFalse(window.local_device_selector.isEnabled())
            window.close()

    def test_window_stops_session_on_switch_rebuild_and_close(self):
        with tempfile.TemporaryDirectory() as folder:
            window = MainWindow(storage_path=Path(folder) / "materials.json")
            window.local_session = session = Mock()
            window.answer_mode.setCurrentIndex(1)
            window.local_device_selector.setCurrentIndex(2)
            session.stop.assert_called_once()
            session.stop.reset_mock()
            window.answer_mode.setCurrentIndex(0)
            session.stop.assert_called_once()
            session.stop.reset_mock()
            window.imported_documents = [self.chunk]
            with patch("desktop_app.IndexWorker"):
                window.on_rebuild()
            session.stop.assert_called_once()
            window.worker = None
            session.stop.reset_mock()
            window.close()
            session.stop.assert_called_once()

    def test_local_answer_display_reports_device_and_layers(self):
        with tempfile.TemporaryDirectory() as folder:
            window = MainWindow(storage_path=Path(folder) / "materials.json")
            self.addCleanup(window.close)
            window.on_answer({"answer": "青竹 [1]", "model": "Qwen3-4B-Q5_K_M (CUDA)",
                              "mode": "local", "usage": None, "citations": [1],
                              "invalid_citations": [], "device": "CUDA", "gpu_layers": 37})
            self.assertIn("实际推理设备：CUDA", window.call_details.toPlainText())
            self.assertIn("GPU 加载层数：37", window.call_details.toPlainText())
            self.assertIn("CUDA", window.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
