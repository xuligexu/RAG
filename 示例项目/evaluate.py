"""文档级检索评测：多个块来自同一文档时合并；不调用聊天模型。"""
import argparse
import json
from pathlib import Path
import sys
import time

from rag import BASE, DEFAULT_URL, Retriever


def document_metrics(hits, relevant_sources, k):
    retrieved = list(dict.fromkeys(hit["source"] for hit in hits))
    relevant = set(relevant_sources)
    if not relevant:
        return None
    found = set(retrieved) & relevant
    first = next((i for i, source in enumerate(retrieved, 1) if source in relevant), None)
    return {"hit": float(bool(found)), "recall": len(found) / len(relevant),
            "reciprocal_rank": 1 / first if first else 0.0,
            "source_precision": len(found) / len(retrieved) if retrieved else 0.0}


def evaluate(index_path=Path("index_bm25.json"),
             questions_path=BASE / "eval_questions.json", mode="bm25",
             top_k=3, candidate_k=10, base_url=DEFAULT_URL):
    """运行文档级检索评测并返回摘要与逐题结果，供编辑器入口或其他 Python 代码调用。"""
    retriever = Retriever.from_file(index_path, base_url)
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    measured, latencies, rows = [], [], []
    for item in questions:
        start = time.perf_counter()
        hits = retriever.search(item["question"], mode, top_k, candidate_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics = document_metrics(hits, item["relevant_sources"], top_k)
        if metrics is not None:
            measured.append(metrics)
        latencies.append(elapsed_ms)
        rows.append({"id": item["id"], "question": item["question"],
                     "sources": list(dict.fromkeys(h["source"] for h in hits)),
                     "metrics": metrics,
                     "note": "无答案题仅展示候选，不计入召回均值" if metrics is None else ""})
    summary = {name: sum(row[name] for row in measured) / len(measured)
               for name in measured[0]} if measured else {}
    summary.update({"mode": mode, "chunk_top_k": top_k,
                    "answerable_count": len(measured), "total_count": len(questions),
                    "mean_retrieval_ms": sum(latencies) / len(latencies) if latencies else 0})
    return {"summary": summary, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("index_bm25.json"))
    parser.add_argument("--questions", type=Path, default=BASE / "eval_questions.json")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    args = parser.parse_args()
    result = evaluate(args.index, args.questions, args.mode, args.top_k,
                      args.candidate_k, args.base_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
