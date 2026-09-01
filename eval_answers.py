"""
RAG 答案级评测脚本（端到端：检索 -> 重排 -> 生成 -> LLM-as-Judge）

与 eval_rag.py（只评测检索质量）互补，本脚本评测"最终回答"的质量：
  1. faithfulness   - 回答中的事实是否都有证据支撑（无幻觉）
  2. relevance      - 回答是否切题、覆盖问题核心

用法：
  python eval_answers.py                          # 使用 eval_questions.json
  python eval_answers.py --questions my_set.json  # 自定义评测集
  python eval_answers.py --top-n 3                # 送入生成的证据块数

说明：
  - 检索/重排复用生产链路（混合检索 + Cross-Encoder 重排）
  - 生成与打分需要 DEEPSEEK_API_KEY（.env 中配置）；缺失时仅输出检索级结果
  - 结果保存到 eval_answers_results.json
"""

import argparse
import json
import os
import sys
import time


def safe_print(text):
    """避免 Windows GBK 控制台对特殊字符报错。"""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all("query" in q for q in data):
        raise ValueError("questions file must be a list of {query, expected, answer}")
    return data


def has_api_key(config):
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def generate_answer(config, docs):
    """使用生产链路（ResponseGenerator）生成带 [n] 引用的回答。"""
    from agents.rag_agent.response_generator import ResponseGenerator
    generator = ResponseGenerator(config)
    result = generator.generate_response(
        query="PLACEHOLDER",
        retrieved_docs=docs,
        picture_paths=[],
        chat_history=None,
    )
    return result


def judge_answer(config, question, answer, evidence_texts):
    """LLM-as-Judge：对 faithful 与 relevance 打分。"""
    evidence = "\n\n".join(f"[{i+1}] {t[:800]}" for i, t in enumerate(evidence_texts))
    prompt = f"""You are an evaluator of a medical QA system. Given the user question, the retrieved evidence chunks, and the system answer:

USER QUESTION: {question["query"]}

EVIDENCE CHUNKS:
{evidence}

SYSTEM ANSWER:
{answer}

Score TWO aspects from 0 to 1:
1. faithfulness: whether every factual claim in the answer is supported by the evidence (no hallucination). 1.0 = fully grounded in evidence, 0.0 = mostly unsupported/fabricated.
2. relevance: whether the answer directly addresses the question and covers its core. 1.0 = fully relevant, 0.0 = off-topic.

Respond with JSON ONLY in this exact format:
{{"faithfulness": 0.0, "relevance": 0.0, "reason": "one short sentence"}}"""
    raw = config.rag.llm.invoke(prompt).content
    try:
        parsed = json.loads(raw)
        return {
            "faithfulness": float(parsed.get("faithfulness", 0.0)),
            "relevance": float(parsed.get("relevance", 0.0)),
            "reason": str(parsed.get("reason", ""))[:300],
        }
    except Exception:
        return {"faithfulness": 0.0, "relevance": 0.0, "reason": "judge parse failed: " + raw[:100]}


def main():
    parser = argparse.ArgumentParser(description="RAG 答案级评测")
    parser.add_argument("--questions", type=str, default="eval_questions.json")
    parser.add_argument("--k", type=int, default=5, help="检索候选数量（默认 5）")
    parser.add_argument("--rerank-top", type=int, default=20, help="重排前保留候选数（默认 20）")
    parser.add_argument("--top-n", type=int, default=3, help="送入生成的证据块数（默认 3）")
    args = parser.parse_args()

    from eval_rag import build_stores, load_questions as _unused, retrieve
    from config import Config

    questions = load_questions(args.questions)
    config = Config()
    dense_store, sparse_store, hybrid_store, reranker = build_stores(config)

    ok_key = has_api_key(config)
    safe_print("=" * 70)
    safe_print("RAG 答案级评测 | 题目数: %d | 证据块 top-%d | LLM 生成+打分: %s"
               % (len(questions), args.top_n, "ON" if ok_key else "OFF(无 API key)"))
    safe_print("=" * 70)

    rows = []
    for i, q in enumerate(questions, start=1):
        raw = retrieve(hybrid_store, q["query"], args.k)
        if reranker is not None and len(raw) > 1:
            candidates = [doc.page_content for doc, _ in raw[:args.rerank_top]]
            scores = reranker.predict([(q["query"], c) for c in candidates])
            pairs = sorted(zip(raw[:args.rerank_top], scores), key=lambda x: x[1], reverse=True)
            top = [pair[0] for pair, _ in pairs][:args.top_n]
        else:
            top = [doc for doc, _ in raw][:args.top_n]

        top_docs = [{
            "id": d.metadata.get("doc_id", ""),
            "content": d.page_content,
            "score": 0.0,
            "source": d.metadata.get("source", ""),
            "source_path": d.metadata.get("source_path", ""),
        } for d in top]

        row = {"query": q["query"], "hit": False, "answer": None,
               "faithfulness": None, "relevance": None, "reason": None}

        # 检索级命中（关键词）
        expected = q.get("expected", "")
        if expected:
            row["hit"] = any(expected.lower() in (d.page_content or "").lower() for d in top)

        if ok_key:
            try:
                result = generate_answer(config, top_docs)
                answer = result["response"]
                row["answer"] = answer[:400]
                judge = judge_answer(config, q, answer, [d.page_content for d, _ in top])
                row["faithfulness"] = judge["faithfulness"]
                row["relevance"] = judge["relevance"]
                row["reason"] = judge["reason"]
            except Exception as exc:
                row["reason"] = "generation error: %s" % str(exc)[:120]

        rows.append(row)
        status = "hit" if row["hit"] else "miss"
        safe_print("[%02d/%02d] %s | %s | faith=%s rel=%s | %s"
                   % (i, len(questions), status, q["query"][:45],
                      row["faithfulness"], row["relevance"], (row["reason"] or "")[:50]))

    # 聚合
    n = len(rows)
    hit_rate = sum(1 for r in rows if r["hit"]) / n if n else 0.0
    f_list = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    r_list = [r["relevance"] for r in rows if r["relevance"] is not None]
    summary = {
        "questions": n,
        "retrieval_hit_rate": round(hit_rate, 4),
        "avg_faithfulness": round(sum(f_list) / len(f_list), 4) if f_list else None,
        "avg_relevance": round(sum(r_list) / len(r_list), 4) if r_list else None,
        "judged": len(f_list),
    }
    safe_print("-" * 70)
    safe_print("汇总: 检索命中率=%.1f%% | 平均 faithfulness=%s | 平均 relevance=%s (已评 %d/%d 题)"
               % (hit_rate * 100, summary["avg_faithfulness"], summary["avg_relevance"], len(f_list), n))

    with open("eval_answers_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": rows}, f, ensure_ascii=False, indent=2)
    safe_print("结果已保存到 eval_answers_results.json")


if __name__ == "__main__":
    main()
