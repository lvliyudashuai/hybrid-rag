"""命令行入口：建库 / 单次问答 / 交互问答。

用法：
    python rag.py --rebuild           # 重建向量库
    python rag.py "你的问题"          # 单次问答（离线模拟或调用 LLM）
    python rag.py                     # 交互式问答
"""
import argparse

import config
from query import generate, retrieve
from vectorstore import is_similarity


def main():
    parser = argparse.ArgumentParser(description="RAG 文档问答助手（CLI）")
    parser.add_argument("query", nargs="*", help="问题文本")
    parser.add_argument("--rebuild", action="store_true", help="重建向量库")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="召回片段数")
    args = parser.parse_args()

    if args.rebuild:
        from ingest import ingest_documents
        n = ingest_documents()
        print(f"✅ 建库完成，共 {n} 个文本块。")
        return

    query_text = " ".join(args.query).strip()
    if query_text:
        answer_single(query_text, args.k)
        return

    interactive(args.k)


def answer_single(query: str, k: int):
    docs = retrieve(query, k=k)
    print(f"\n🔍 问题：{query}\n")
    if not docs:
        print("未检索到相关文档。")
        return
    for i, (doc, score) in enumerate(docs, 1):
        section = doc.metadata.get("section", "未知章节")
        sim = is_similarity(score)
        print(f"--- 片段 {i} | {section} | 相似度 {sim:.1%} ---")
        print(doc.page_content[:300])
        print()
    print("💬 生成回答：\n")
    print(generate(query, [d for d, _ in docs]))


def interactive(k: int):
    print("🔍 RAG 问答系统已启动，输入 'exit' 退出。")
    while True:
        try:
            query = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if query.lower() in {"exit", "quit"}:
            print("再见！")
            break
        if not query:
            continue
        answer_single(query, k)


if __name__ == "__main__":
    main()
