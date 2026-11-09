"""编辑器一键运行入口：修改下方配置，再用编辑器的运行按钮执行此文件。"""
import json
from pathlib import Path
import unittest

import evaluate
import rag


BASE = Path(__file__).resolve().parent

# 可选任务："bm25_demo"、"build"、"search"、"evaluate"、"ask"、"tests"
TASK = "bm25_demo"

# 按需修改下方参数。BM25 不需要 Ollama 或模型；dense/hybrid/ask 需要本地模型服务。
MODE = "bm25"  # "bm25"、"dense" 或 "hybrid"
DATA_PATH = BASE / "data"
INDEX_PATH = BASE / "index_editor_bm25.json"
QUESTION = "退款申请需要满足哪些条件"
TOP_K = 3
CANDIDATE_K = 10
CHUNK_SIZE = 600
OVERLAP = 100
EMBEDDING_MODEL = "bge-m3"
CHAT_MODEL = "qwen2.5:3b"
BASE_URL = rag.DEFAULT_URL


def show(title, value):
    print(f"\n=== {title} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run():
    if TASK == "bm25_demo":
        index = rag.build_index(DATA_PATH, INDEX_PATH, mode="bm25",
                                size=CHUNK_SIZE, overlap=OVERLAP,
                                embedding_model=EMBEDDING_MODEL, 
                                base_url=BASE_URL
                            )
        show("建库完成", {
                    "documents": index["document_count"],
                    "chunks": len(index["chunks"]),
                    "index": str(INDEX_PATH),
                }
            )
        retriever = rag.Retriever(index, BASE_URL)
        hits = retriever.search(QUESTION, mode="bm25", top_k=TOP_K,
                                candidate_k=CANDIDATE_K)
        show("检索结果", hits)
        result = evaluate.evaluate(INDEX_PATH, BASE / "eval_questions.json",
                                   mode="bm25", top_k=TOP_K,
                                   candidate_k=CANDIDATE_K, base_url=BASE_URL)
        show("检索评测", result)
        return

    if TASK == "build":
        index = rag.build_index(DATA_PATH, INDEX_PATH, mode=MODE,
                                size=CHUNK_SIZE, overlap=OVERLAP,
                                embedding_model=EMBEDDING_MODEL, base_url=BASE_URL)
        show("建库完成", {
            "documents": index["document_count"],
            "chunks": len(index["chunks"]),
            "dimension": index["dimension"],
            "index": str(INDEX_PATH),
        })
        return

    if TASK == "search":
        retriever = rag.Retriever.from_file(INDEX_PATH, BASE_URL)
        show("检索结果", retriever.search(QUESTION, mode=MODE, top_k=TOP_K,
                                          candidate_k=CANDIDATE_K))
        return

    if TASK == "evaluate":
        show("检索评测", evaluate.evaluate(
            INDEX_PATH, BASE / "eval_questions.json", mode=MODE,
            top_k=TOP_K, candidate_k=CANDIDATE_K, base_url=BASE_URL))
        return

    if TASK == "ask":
        retriever = rag.Retriever.from_file(INDEX_PATH, BASE_URL)
        hits = retriever.search(QUESTION, mode=MODE, top_k=TOP_K,
                                candidate_k=CANDIDATE_K)
        show("问答结果", rag.generate_answer(QUESTION, hits, CHAT_MODEL, BASE_URL))
        return

    if TASK == "tests":
        suite = unittest.defaultTestLoader.discover(str(BASE), pattern="test_rag.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print(f"\n测试结果：通过 {result.testsRun - len(result.failures) - len(result.errors)}，"
              f"失败 {len(result.failures)}，错误 {len(result.errors)}")
        return

    raise ValueError(f"未知 TASK：{TASK}")


if __name__ == "__main__":
    run()
