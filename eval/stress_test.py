"""Offline D2L capacity smoke test; never opens the user's material library."""

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CASES = [
    ("bias", "线性回归中的偏置项表示什么？",
     "chapter_linear-networks/linear-regression.md",
     "偏置是指当所有特征都取值为0时，预测值应该为多少。"),
    ("batch", "小批量随机梯度下降的批量大小和学习率是训练得到的还是预先指定的？",
     "chapter_linear-networks/linear-regression.md",
     "批量大小和学习率的值通常是手动预先指定，而不是通过模型训练得到的。"),
    ("dropout", "神经网络在测试时通常还需要使用暂退法、丢弃节点吗？",
     "chapter_multilayer-perceptrons/dropout.md", "通常，我们在测试时不用暂退法。"),
    ("pooling", "最大汇聚和平均汇聚计算什么，汇聚层有可学习参数吗？",
     "chapter_convolutional-neural-networks/pooling.md", "汇聚层不包含参数。"),
    ("clip", "梯度裁剪可以防止梯度爆炸吗？它能应对梯度消失吗？",
     "chapter_recurrent-neural-networks/rnn-scratch.md",
     "梯度裁剪可以防止梯度爆炸，但不能应对梯度消失。"),
    ("attention", "查询和键的矢量长度不同时，适合使用哪种注意力评分函数？",
     "chapter_attention-mechanisms/attention-scoring-functions.md",
     "当查询和键是不同长度的矢量时，可以使用可加性注意力评分函数。"),
    ("position", "自注意力为什么需要添加位置编码？",
     "chapter_attention-mechanisms/self-attention-and-positional-encoding.md",
     "来注入绝对的或相对的位置信息。"),
    ("decay", "权重衰减怎样调整函数的复杂度？",
     "chapter_multilayer-perceptrons/weight-decay.md",
     "权重衰减为我们提供了一种连续的机制来调整函数的复杂度。"),
]


class MemorySampler:
    """Optional sampled parent-process RSS, not total system/llama memory."""

    def __init__(self):
        self.peak = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            return self

        def sample():
            while not self._stop.is_set():
                self.peak = max(self.peak or 0, process.memory_info().rss)
                self._stop.wait(0.1)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join()


def desktop_boundaries(dataset, paths, output):
    from PySide6.QtWidgets import QApplication
    from desktop_app import MainWindow
    from material_store import load_materials
    from read_document import MAX_MATERIAL_FILES

    app = QApplication.instance() or QApplication([])
    result = {}
    window = MainWindow(storage_path=output / "desktop" / "materials.json")
    try:
        window.import_files([str(p) for p in paths[:11]])
        result["eleven_file_batch"] = {
            "accepted": len(window.imported_documents),
            "message": window.import_errors.toPlainText(),
        }
        before = len(window.imported_documents)
        window.import_files([str(paths[0])] * (MAX_MATERIAL_FILES + 1))
        result["oversized_batch"] = {
            "previous_count": before, "accepted_total": len(window.imported_documents),
            "message": window.import_errors.toPlainText(),
        }
    finally:
        window.close()
    duplicate_paths = [p for p in paths if p.name == "index.md"][:2]
    window = MainWindow(storage_path=output / "duplicates" / "materials.json")
    try:
        window.import_files([str(p) for p in duplicate_paths])
        result["same_basename_different_directories"] = {
            "selected": len(duplicate_paths),
            "accepted": len(window.imported_documents),
            "message": window.import_errors.toPlainText(),
        }
    finally:
        window.close()
    window = MainWindow(storage_path=output / "folder" / "materials.json")
    try:
        window.import_folder(dataset)
        result["recursive_folder"] = {
            "accepted": len(window.imported_documents),
            "message": window.import_errors.toPlainText(),
            "restored_count": len(load_materials(window.storage_path)),
        }
    finally:
        window.close()
    storage = output / "full_desktop" / "materials.json"
    window = MainWindow(storage_path=storage)
    try:
        window.import_files(paths)
        names = [d.metadata["source"] for d in window.imported_documents]
        result["canonical_collection"] = {
            "selected": len(paths), "accepted": len(names),
            "unique_source_names": len({n.casefold() for n in names}),
            "message": window.import_errors.toPlainText(),
        }
    finally:
        window.close()
    window = MainWindow(storage_path=storage)
    try:
        documents = list(window.imported_documents)
        result["canonical_collection"]["restored_count"] = len(documents)
        result["canonical_collection"]["source_names_preserved"] = names == [d.metadata["source"] for d in documents]
        window.import_files(paths)
        result["repeat_import_count"] = len(window.imported_documents)
    finally:
        window.close()
    if len(documents) != len(paths) or len(set(names)) != len(paths):
        raise ValueError("Actual desktop import did not preserve the complete collection.")
    # Keep the application alive throughout the widget checks.
    assert app is not None
    return result, documents


def write_report(output, report):
    target = output / "report.json"
    temporary = output / "report.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--index", action="store_true", help="Actually encode with local BGE-M3.")
    parser.add_argument("--local-answers", action="store_true", help="Three offline Qwen3 calls.")
    args = parser.parse_args()
    if args.local_answers and not args.index:
        parser.error("--local-answers requires --index")
    root = Path(__file__).resolve().parents[1]
    dataset = args.dataset.resolve(strict=True)
    now = datetime.now(timezone(timedelta(hours=8)))
    output = root / "eval" / "stress_results" / now.strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started": now.isoformat(), "dataset": str(dataset),
              "output": str(output), "source_repository": "https://github.com/d2l-ai/d2l-zh",
              "commit": None, "paid_api_calls": 0,
              "scope": "Actual desktop import/persistence/index/cache smoke test",
              "settings": {"model": "bge-m3", "chunk_size": 500, "overlap": 80,
                           "batch_size": 8, "top_k": 3, "query_repeats": 3},
              "limitations": [
                  "Downloaded ZIP has no verified commit; file hashes identify this local snapshot.",
                  "Uses real desktop disambiguated source labels, not the old backend-only relative labels.",
                  "Eight development probes are not an independent final accuracy benchmark.",
                  "Exact reference-quote hit is strict; equivalent evidence may count as a miss.",
                  "No GB-scale single file, Office-heavy collection, or interactive UI latency test.",
                  "RSS sampled for parent process only; PyTorch peak excludes llama and other apps.",
              ]}
    try:
        from read_document import load_imported_text
        from split_documents import split_documents
        from embed_query import retrieve
        import torch
        from PySide6.QtWidgets import QApplication
        from desktop_app import MainWindow, IndexWorker

        app = QApplication.instance() or QApplication([])

        report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                 "packages": {p: version(p) for p in [
                                     "torch", "sentence-transformers", "transformers",
                                     "langchain-text-splitters", "PySide6"]}}
        all_md = sorted(dataset.rglob("*.md"))
        paths = [p for p in all_md if p.relative_to(dataset).parts[0].startswith("chapter_")
                 and not p.stem.endswith("_origin")]
        if not paths:
            raise ValueError("No canonical chapter Markdown files found.")
        report["licenses"] = {
            "root_LICENSE": (dataset / "LICENSE").read_text(encoding="utf-8"),
            "book_config_copyright": next(
                line for line in (dataset / "config.ini").read_text(encoding="utf-8").splitlines()
                if line.startswith("copyright")),
            "distribution": "Local testing only; dataset excluded from this project's Git submissions.",
        }
        started = time.perf_counter()
        documents, manifest = [], []
        for path in paths:
            doc = load_imported_text(path)
            relative = path.relative_to(dataset).as_posix()
            doc.metadata["source"] = relative
            documents.append(doc)
            manifest.append({"path": relative, "bytes": path.stat().st_size,
                             "characters": len(doc.page_content),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        report["manifest"] = manifest
        report["dataset_stats"] = {
            "all_markdown": len(all_md), "canonical_chapters": len(documents),
            "raw_bytes": sum(m["bytes"] for m in manifest),
            "characters": sum(m["characters"] for m in manifest),
            "largest_file_bytes": max(m["bytes"] for m in manifest),
            "duplicate_basenames": {n: c for n, c in Counter(p.name for p in paths).items() if c > 1},
            "read_seconds": time.perf_counter() - started,
        }
        started = time.perf_counter()
        chunks = split_documents(documents)
        text_by_source = {d.metadata["source"]: d.page_content for d in documents}
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            if text_by_source[chunk.metadata["source"]][start:start + len(chunk.page_content)] != chunk.page_content:
                raise ValueError("Chunk offset does not match unchanged source text.")
        report["split"] = {"chunks": len(chunks), "max_characters": max(len(c.page_content) for c in chunks),
                           "seconds": time.perf_counter() - started, "all_offsets_valid": True}
        report["desktop"], documents = desktop_boundaries(dataset, paths, output)
        source_map = {path.relative_to(dataset).as_posix(): doc.metadata["source"]
                      for path, doc in zip(paths, documents)}
        print("DATASET:", report["dataset_stats"], flush=True)
        print("SPLIT:", report["split"], flush=True)
        print("DESKTOP:", report["desktop"], flush=True)
        for _, _, source, quote in CASES:
            if quote not in text_by_source[source]:
                raise ValueError(f"Reference quote missing from {source}")
        write_report(output, report)
        if args.index:
            cache = output / "full_desktop" / "index.npz"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            print("Building BGE-M3 index once on the full collection...", flush=True)
            started = time.perf_counter()
            with MemorySampler() as memory:
                built, errors = [], []
                worker = IndexWorker("bge-m3", documents, cache_path=cache)
                worker.succeeded.connect(lambda value, _: built.append(value))
                worker.failed.connect(errors.append)
                worker.run()
                if not built:
                    raise RuntimeError(errors[0] if errors else "Desktop IndexWorker returned no index.")
                index = built.pop()
                del worker
                if index[2].is_cuda:
                    torch.cuda.synchronize()
            model, chunks, vectors, prefixes = index
            report["index"] = {"seconds": time.perf_counter() - started,
                               "cache_status": model._project_index_cache_status,
                               "shape": list(vectors.shape), "device": str(vectors.device),
                               "vector_bytes": vectors.numel() * vectors.element_size(),
                               "cache_bytes": cache.stat().st_size if cache.exists() else None,
                               "sampled_parent_peak_rss_bytes": memory.peak,
                               "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                               if vectors.is_cuda else None}
            report["probes"] = []
            for case_id, question, source, quote in CASES:
                source = source_map[source]
                durations = []
                for _ in range(3):
                    started = time.perf_counter()
                    hits = retrieve(question, model, chunks, vectors, prefixes, k=3)
                    if vectors.is_cuda:
                        torch.cuda.synchronize()
                    durations.append(time.perf_counter() - started)
                probe = {"id": case_id, "question": question, "reference_source": source,
                         "reference_quote": quote, "seconds": durations,
                         "median_seconds": statistics.median(durations),
                         "reference_quote_ranks": [rank for rank, (_, c, _) in enumerate(hits, 1)
                                                   if c.metadata["source"] == source and quote in c.page_content],
                         "reference_document_ranks": [rank for rank, (_, c, _) in enumerate(hits, 1)
                                                      if c.metadata["source"] == source],
                         "hits": [{"rank": rank, "chunk": number, "source": c.metadata["source"],
                                   "start_index": c.metadata["start_index"], "score": score,
                                   "text": c.page_content}
                                  for rank, (number, c, score) in enumerate(hits, 1)]}
                report["probes"].append(probe)
                print("PROBE:", case_id, "quote ranks:", probe["reference_quote_ranks"],
                      "median seconds:", round(probe["median_seconds"], 4), flush=True)
            report["retrieval_summary"] = {
                "questions": len(CASES),
                "top3_reference_quote_hits": sum(bool(p["reference_quote_ranks"]) for p in report["probes"]),
                "top3_reference_document_hits": sum(bool(p["reference_document_ranks"]) for p in report["probes"]),
                "median_query_seconds": statistics.median(p["median_seconds"] for p in report["probes"]),
            }
            write_report(output, report)
            if args.local_answers:
                from local_answer import LocalAnswerSession
                from langchain_core.documents import Document
                session = LocalAnswerSession()
                report["local_answers"] = []
                try:
                    for case_id in ("bias", "dropout", "clip"):
                        probe = next(p for p in report["probes"] if p["id"] == case_id)
                        selected = [Document(page_content=h["text"],
                                             metadata={"source": h["source"], "start_index": h["start_index"]})
                                    for h in probe["hits"]]
                        started = time.perf_counter()
                        try:
                            response = session.answer(probe["question"], selected, device="auto")
                            item = {"id": case_id, "seconds": time.perf_counter() - started,
                                    "pid": session.pid, "response": response,
                                    "semantic_support": "requires manual review"}
                        except Exception as exc:
                            item = {"id": case_id, "seconds": time.perf_counter() - started,
                                    "error_type": type(exc).__name__, "message": str(exc)}
                        report["local_answers"].append(item)
                        print("LOCAL:", item, flush=True)
                        write_report(output, report)
                finally:
                    session.stop()
                    report["local_session_stopped"] = session.pid is None
            expected = vectors.detach().cpu().clone()
            del index, model, vectors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            started = time.perf_counter()
            window = MainWindow(storage_path=output / "full_desktop" / "materials.json")
            try:
                # Exercise the actual startup IndexWorker and queued UI callbacks.
                while window.worker is not None:
                    app.processEvents()
                    time.sleep(0.01)
                if window.index is None:
                    raise RuntimeError("Desktop startup did not restore its index.")
                restored = window.index
                window.resize(1100, 760)
                window.result_tabs.setCurrentIndex(2)
                window.show()
                app.processEvents()
                window.grab().save(str(output / "desktop_ready.png"))
                startup_status = window.import_status.text()
                query_enabled = window.query_button.isEnabled()
            finally:
                window.close()
            if restored[2].is_cuda:
                torch.cuda.synchronize()
            report["cache_restore"] = {
                "seconds": time.perf_counter() - started,
                "status": restored[0]._project_index_cache_status,
                "vectors_exactly_equal": bool(torch.equal(expected, restored[2].detach().cpu())),
                "desktop_status": startup_status,
                "query_enabled": query_enabled,
            }
            del restored
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_report(output, report)
        print("REPORT:", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
