"""验证算法边界和数据流，不代表真实模型效果评测。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rag
from evaluate import document_metrics


class RagTests(unittest.TestCase):
    def test_window_coverage_and_tail(self):
        text = "0123456789abcdef"
        windows = list(rag.split_windows(text, 6, 2))
        self.assertEqual([(a, b) for a, b, _ in windows], [(0, 6), (4, 10), (8, 14), (12, 16)])
        covered = {i for a, b, _ in windows for i in range(a, b)}
        self.assertEqual(covered, set(range(len(text))))
        self.assertEqual(list(rag.split_windows("", 6, 2)), [])
        self.assertEqual(len(list(rag.split_windows("abc", 6, 2))), 1)
        with self.assertRaises(ValueError):
            list(rag.split_windows(text, 6, 6))

    def test_vector_math_and_bad_vectors(self):
        self.assertAlmostEqual(rag.dot(rag.normalize([1, 0]), rag.normalize([1, 1])), 2 ** -0.5)
        for bad in [[], [0, 0], [float("nan")], [float("inf")]]:
            with self.assertRaises(ValueError):
                rag.normalize(bad)
        with self.assertRaises(ValueError):
            rag.dot([1], [1, 2])

    def test_rrf_worked_example(self):
        scores = rag.rrf([["A", "B", "C"], ["B", "D", "A"]])
        self.assertEqual(sorted(scores, key=scores.get, reverse=True), ["B", "A", "D", "C"])

    def test_bm25_identifier_and_chinese(self):
        bm25 = rag.BM25(["E104 会话过期", "E205 网络超时", "退款条件"])
        self.assertEqual(rag.top_indices(bm25.scores("E104"), 1), [0])
        self.assertEqual(rag.top_indices(bm25.scores("退款"), 1), [2])
        self.assertEqual(rag.top_indices(bm25.scores("xyz999"), 3, True), [])

    def test_build_search_and_rebuild_removes_deleted_document(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data"
            data.mkdir()
            (data / "a.md").write_text("# 退款\n购买后7天内申请。", encoding="utf-8")
            second = data / "b.md"
            second.write_text("# 网络\nE205 表示网络超时。", encoding="utf-8")
            target = root / "index.json"
            rag.build_index(data, target)
            retriever = rag.Retriever.from_file(target)
            self.assertEqual(retriever.search("E205")[0]["source"], "b.md")
            second.unlink()
            rag.build_index(data, target)
            self.assertEqual(rag.Retriever.from_file(target).search("E205"), [])

    def test_mock_dense_to_generation_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "doc.md").write_text("# 退款\n购买后7天内申请。", encoding="utf-8")
            def fake_post(base, endpoint, payload):
                if endpoint == "/api/embed":
                    self.assertFalse(payload["truncate"])
                    return {"embeddings": [[1, 0] for _ in payload["input"]]}
                self.assertFalse(payload["stream"])
                evidence = json.loads(payload["messages"][1]["content"])["evidence"]
                self.assertEqual(evidence[0]["source"], "doc.md")
                return {"message": {"content": "购买后7天内可以申请。[S1]"}}
            with patch("rag.post_json", side_effect=fake_post):
                index = rag.build_index(root, root / "index.json", mode="hybrid")
                retriever = rag.Retriever(index)
                hits = retriever.search("退款", mode="hybrid")
                result = rag.generate_answer("退款", hits)
                self.assertEqual(result["warnings"], [])
                self.assertIn("[S1]", result["answer"])

    def test_embedding_count_and_missing_dense_index(self):
        with patch("rag.post_json", return_value={"embeddings": []}):
            with self.assertRaises(ValueError):
                rag.embed(["问题"], "fake")
        chunks, _ = rag.load_chunks(rag.BASE / "data")
        retriever = rag.Retriever({"schema_version": 1, "chunks": chunks, "vectors": None})
        with self.assertRaises(ValueError):
            retriever.search("问题", mode="dense")
        with self.assertRaises(ValueError):
            retriever.search("问题", top_k=11, candidate_k=10)

    def test_context_budget_citations_and_empty_refusal(self):
        hit = {"source":"a.md", "heading":"标题", "text":"证据", "start":0, "end":2}
        self.assertEqual(rag.pack_context([hit], max_chars=1), [])
        evidence = rag.pack_context([hit])
        self.assertTrue(rag.citation_warnings("事实。[S9]", evidence))
        with patch("rag.post_json") as request:
            result = rag.generate_answer("问题", [])
            request.assert_not_called()
            self.assertIn("无法确定", result["answer"])

    def test_document_metric_dedup_and_unanswerable(self):
        hits = [{"source":"x"}, {"source":"x"}, {"source":"a"}]
        metrics = document_metrics(hits, ["a", "b"], 3)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["source_precision"], 0.5)
        self.assertIsNone(document_metrics(hits, [], 3))


if __name__ == "__main__":
    unittest.main()
