"""
RAG 检索策略评测脚本

对比三种检索策略在医学问答上的效果：
  1. DENSE   - 纯稠密向量检索（语义匹配）
  2. SPARSE  - 纯稀疏向量检索（BM25 关键词匹配）
  3. HYBRID  - 稠密 + 稀疏混合检索（Qdrant RRF 融合）
  4. HYBRID+ - 混合检索 + Cross-Encoder 重排序（生产完整链路）

用法：
  python eval_rag.py                     # 使用内置评测集
  python eval_rag.py --questions q.json  # 使用自定义评测集
  python eval_rag.py --k 5 --top 20      # 调整评测参数

输出：各策略的 Hit@k / MRR 对比表，以及每道题的命中明细。
不依赖任何 LLM API，仅使用本地嵌入模型与重排序模型，可离线运行。

评测集 JSON 格式：
  [{"query": "问题文本", "expected": "期望命中的关键词（出现在目标文档块中）"}, ...]
"""

import argparse
import json
import sys
import time


DEFAULT_QUESTIONS = [
    {"query": "What are the types of brain tumors?",
     "expected": "Types of brain tumors"},
    {"query": "What causes brain tumors?",
     "expected": "What causes brain tumors?"},
    {"query": "Which imaging tests are used to diagnose brain tumors?",
     "expected": "Imaging tests"},
    {"query": "How is radiation therapy used to treat brain tumors?",
     "expected": "Radiation"},
    {"query": "What are the common symptoms of brain tumors?",
     "expected": "What are the symptoms?"},
    {"query": "What treatments are available for brain tumors?",
     "expected": "What treatments are available?"},
    {"query": "How does COVID-19 present in chest X-ray images?",
     "expected": "COVID-19 detection from chest X-ray"},
    {"query": "Why is image enhancement important before X-ray classification?",
     "expected": "Image enhancement"},
    {"query": "What is transfer learning and how is it applied to medical imaging?",
     "expected": "Transfer learning"},
    {"query": "What is the star-shape loss used for in skin lesion segmentation?",
     "expected": "Star-Shape Loss"},
    {"query": "Which loss functions are commonly used for medical image segmentation?",
     "expected": "3.2. Loss Functions"},
    {"query": "How does deep learning differ from traditional machine learning?",
     "expected": "Deep learning"},
]


def load_questions(path):
    """Load a custom question set from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(
        "query" in q and "expected" in q for q in data
    ):
        raise ValueError("questions file must be a list of {query, expected}")
    return data


def check_dependencies():
    """Verify required packages are importable, with friendly error messages."""
    missing = []
    try:
        from langchain_qdrant import QdrantVectorStore, RetrievalMode  # noqa
    except ImportError:
        missing.append("langchain-qdrant")
    try:
        from sentence_transformers import CrossEncoder  # noqa
    except ImportError:
        missing.append("sentence-transformers")
    try:
        from qdrant_client import QdrantClient  # noqa
    except ImportError:
        missing.append("qdrant-client")
    if missing:
        print("缺少依赖: " + ", ".join(missing))
        print("请先安装: pip install " + " ".join(missing))
        sys.exit(1)


def build_stores(config):
    """
    构建三种检索模式的向量库实例。

    返回 (dense_store, sparse_store, hybrid_store, reranker)
    """
    from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder

    client = QdrantClient(path=config.rag.vector_local_path)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    common = dict(
        client=client,
        collection_name=config.rag.collection_name,
        embedding=config.rag.embedding_model,
        sparse_embedding=sparse_embeddings,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    dense_store = QdrantVectorStore(
        retrieval_mode=RetrievalMode.DENSE, **common
    )
    sparse_store = QdrantVectorStore(
        retrieval_mode=RetrievalMode.SPARSE, **common
    )
    hybrid_store = QdrantVectorStore(
        retrieval_mode=RetrievalMode.HYBRID, **common
    )

    reranker = CrossEncoder(config.rag.reranker_model)
    return dense_store, sparse_store, hybrid_store, reranker


def get_all_documents(store):
    """
    获取知识库中所有文档块的 page_content 列表。
    使用 Qdrant 的 scroll 接口遍历集合。
    """
    client = store.client
    collection_name = store.collection_name

    all_docs = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            # LangChain 通常将页面内容存储在 payload 的 "page_content" 字段
            page_content = point.payload.get("page_content", "")
            all_docs.append(page_content)
        if next_page is None:
            break
    return all_docs


def retrieve(store, query, k):
    """Run similarity search with score, returning list of (doc, score)."""
    try:
        return store.similarity_search_with_score(query=query, k=k)
    except Exception as exc:
        print("检索失败: %s (%s)" % (query, exc))
        return []


def is_hit(doc, expected):
    """判断检索结果是否命中期望内容（按内容关键词匹配）。"""
    text = doc.page_content
    return expected.lower() in text.lower()


def evaluate(store, reranker, questions, k, rerank_top, all_docs):
    """
    对评测集执行检索，计算 Hit@k、MRR、Precision@k 和 Recall@k。

    all_docs: 知识库中所有文档块的 page_content 列表，用于计算 Recall 的分母。
    """
    stats = {
        "hit_at_1": 0,
        "hit_at_k": 0,
        "mrr": 0.0,
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
    }
    details = []

    for q in questions:
        query = q["query"]
        expected = q["expected"]

        raw = retrieve(store, query, k)

        # 重排序：取前 rerank_top 个候选，用 Cross-Encoder 打分排序
        if reranker is not None and len(raw) > 1:
            candidates = [doc.page_content for doc, _ in raw[:rerank_top]]
            scores = reranker.predict([(query, c) for c in candidates])
            pairs = sorted(zip(raw[:rerank_top], scores), key=lambda x: x[1], reverse=True)
            ordered = [pair[0] for pair, _ in pairs]
        else:
            ordered = [doc for doc, _ in raw]

        # 原有命中位置计算
        hit_pos = None
        for i, doc in enumerate(ordered):
            if is_hit(doc, expected):
                hit_pos = i + 1
                break

        if hit_pos is not None:
            if hit_pos == 1:
                stats["hit_at_1"] += 1
            if hit_pos <= k:
                stats["hit_at_k"] += 1
            stats["mrr"] += 1.0 / hit_pos

        # 新增：Precision@k 与 Recall@k
        # 统计前 k 个结果中命中的文档数量
        top_k_docs = ordered[:k]
        relevant_in_topk = sum(1 for doc in top_k_docs if is_hit(doc, expected))
        precision = relevant_in_topk / k if k > 0 else 0.0

        # 计算知识库中总相关文档数（所有包含 expected 关键词的文档）
        total_relevant = sum(
            1 for doc_text in all_docs if expected.lower() in doc_text.lower()
        )
        recall = relevant_in_topk / total_relevant if total_relevant > 0 else 0.0

        stats["precision_at_k"] += precision
        stats["recall_at_k"] += recall

        details.append({
            "query": query,
            "expected": expected,
            "hit_position": hit_pos,
            "matched_chunk": (
                ordered[hit_pos - 1].page_content[:120]
                if hit_pos is not None else None
            ),
            "precision_at_k": precision,
            "recall_at_k": recall,
        })

    n = len(questions)
    stats["hit_at_1"] /= n
    stats["hit_at_k"] /= n
    stats["mrr"] /= n
    stats["precision_at_k"] /= n
    stats["recall_at_k"] /= n
    return stats, details


def main():
    # Windows GBK 控制台对特殊字符（如数学符号）会报错，容错处理
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="RAG 检索策略评测")
    parser.add_argument("--questions", type=str, default=None,
                        help="评测集 JSON 文件路径（默认使用内置评测集）")
    parser.add_argument("--k", type=int, default=5,
                        help="检索候选数量 k（默认 5）")
    parser.add_argument("--rerank-top", type=int, default=20,
                        help="重排序前保留的候选数（默认 20）")
    args = parser.parse_args()

    check_dependencies()

    from config import Config

    questions = (
        load_questions(args.questions)
        if args.questions else DEFAULT_QUESTIONS
    )
    print("=" * 70)
    print("RAG 检索策略评测 | 评测题数: %d | 候选 k=%d | 重排前 %d"
          % (len(questions), args.k, args.rerank_top))
    print("=" * 70)

    config = Config()
    dense_store, sparse_store, hybrid_store, reranker = build_stores(config)

    # 获取所有文档内容，用于计算 Recall
    all_docs = get_all_documents(dense_store)

    strategies = {
        "DENSE 纯稠密": (dense_store, None),
        "SPARSE 纯BM25": (sparse_store, None),
        "HYBRID 混合": (hybrid_store, None),
        "HYBRID+重排": (hybrid_store, reranker),
    }

    print()
    header = "%-16s %10s %10s %10s %10s %10s %8s" % (
        "策略", "Hit@1", "Hit@%d" % args.k, "MRR", "Prec@%d" % args.k, "Rec@%d" % args.k, "耗时")
    print(header)
    print("-" * len(header))

    results = {}
    for name, (store, reranker_) in strategies.items():
        start = time.time()
        stats, details = evaluate(
            store, reranker_, questions, args.k, args.rerank_top, all_docs
        )
        elapsed = time.time() - start
        results[name] = {"stats": stats, "details": details}
        print("%-16s %10.1f%% %10.1f%% %10.4f %10.1f%% %10.1f%% %7.1fs"
              % (name,
                 stats["hit_at_1"] * 100,
                 stats["hit_at_k"] * 100,
                 stats["mrr"],
                 stats["precision_at_k"] * 100,
                 stats["recall_at_k"] * 100,
                 elapsed))

    # 保存完整结果（先保存，避免后续打印异常导致结果丢失）
    out = "eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print()
    print("完整结果已保存到 %s" % out)

    # 输出每道题的命中明细（仅混合+重排）
    print()
    print("-" * 70)
    print("命中明细（HYBRID+重排 策略）")
    print("-" * 70)
    for d in results["HYBRID+重排"]["details"]:
        status = "HIT@%d" % d["hit_position"] if d["hit_position"] else "MISS"
        snippet = (d["matched_chunk"] or "")[:80].replace("\n", " ")
        print("[%s] %s  =>  %s" % (status, d["query"][:40], snippet))


if __name__ == "__main__":
    main()