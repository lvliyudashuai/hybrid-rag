"""命令行入口：建库 / 问答 / 巡检。

常用：
    python rag.py --doctor                  # 环境自检（第一次跑先看这个）
    python rag.py --rebuild                 # 重建索引
    python rag.py "文档里讲了什么"           # 单次问答
    python rag.py                           # 交互式问答
    python rag.py --ingest ./docs --append  # 追加一个目录（增量，不动已有文档）
    python rag.py --stats                   # 看库里有什么
    python rag.py --prune                   # 清理向量库残留目录
"""
from __future__ import annotations

import argparse
import json
import sys

import config
import logging_config


def cmd_doctor() -> int:
    """环境自检：一次性回答"为什么跑不起来"这类问题。"""
    import vectorstore

    print(f"rag_demo v{config.__version__}")
    print(f"  配置文件      : {config.ENV_FILE}（{'存在' if config.ENV_FILE.exists() else '不存在'}）")
    print(f"  Python        : {sys.version.split()[0]} @ {sys.executable}")

    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无（用 CPU）"
        print(f"  torch         : {torch.__version__} / 设备 {vectorstore.resolve_device()} / GPU {gpu}")
    except Exception as exc:
        print(f"  torch         : 不可用（{exc}）")

    print(f"  embedding     : {config.EMBEDDING_MODEL}"
          f"（本地模式={'开' if config.EMBEDDING_LOCAL_ONLY else '关'}）")
    print(f"  检索          : 模式={config.RETRIEVAL_MODE} top_k={config.TOP_K}"
          f" 候选={config.CANDIDATE_K} alpha={config.HYBRID_ALPHA}"
          f" MMR={'开' if config.USE_MMR else '关'}"
          f" 重排={'开' if config.USE_RERANK else '关'}")

    provider = config.LLM_PROVIDER
    key = config.api_key(provider)
    source = "环境变量" if config.env_api_key(provider) else "界面/无"
    print(f"  生成模型      : {provider} / {config.LLM_MODEL}"
          f" / Key {config.mask_secret(key)}（来源：{source}）")
    if provider != "mock" and not key:
        print("     ⚠️  没有 Key，会自动退化成离线模拟；想真实生成请填 DEEPSEEK_API_KEY。")

    stats = vectorstore.collection_stats()
    print(f"  向量库        : {stats['directory']}")
    print(f"    集合 {stats['collection']}：{'✓ ' + str(stats['count']) + ' 块' if stats['exists'] else '✗ 尚未建立'}")
    for item in vectorstore.list_collections():
        print(f"      - {item['name']}: {item['count']} 块")

    try:
        import jieba  # noqa: F401

        print("  分词          : jieba（中文分词已启用）")
    except ImportError:
        print("  分词          : 未装 jieba，中文退化为二元组（pip install jieba 可改善）")
    return 0


def cmd_stats() -> int:
    import vectorstore

    stats = vectorstore.collection_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\n所有集合：")
    for item in vectorstore.list_collections():
        print(f"  - {item['name']}: {item['count']} 块")
    orphans = vectorstore.prune_orphan_segments(dry_run=True)
    if orphans:
        print(f"\n发现 {len(orphans)} 个孤儿段目录（可执行 --prune 清理）")
    return 0


def cmd_prune() -> int:
    import vectorstore

    orphans = vectorstore.prune_orphan_segments(dry_run=True)
    if not orphans:
        print("没有需要清理的残留目录。")
        return 0
    print(f"将删除 {len(orphans)} 个目录：{', '.join(orphans)}")
    vectorstore.prune_orphan_segments(dry_run=False)
    print("✅ 清理完成。")
    return 0


def cmd_ingest(spec: str, append: bool) -> int:
    import ingest

    report = ingest.ingest_sources(spec, reset=not append,
                                   doc_type=config.DOC_TYPE if not append else "")
    for item in report.files:
        mark = "✅" if item.ok else "❌"
        detail = f"{item.chunks} 块" if item.ok else item.error
        print(f"  {mark} {item.path}  {detail}")
    print(report.summary())
    return 0 if report.chunks else 1


def _print_hits(hits) -> None:
    for hit in hits:
        score = (
            f"相似度 {hit.similarity:.1%}" if hit.similarity is not None
            else f"BM25 {hit.keyword_score:.1f}" if hit.keyword_score is not None
            else f"{hit.score:.4f}"
        )
        print(f"  [{hit.rank}] {hit.locator} ｜ {score}")
        print(f"      {hit.preview(160)}")


def answer_once(question: str, args, history=None) -> int:
    import query
    from retrieval import Trace

    trace = Trace() if args.trace else None
    result = query.ask(question, args.k, history=history, provider=args.provider,
                       trace=trace, mode=args.mode)
    print(f"\n🔍 问题：{question}")
    print(f"\n📎 检索到的片段（{len(result.hits)} 条）：")
    _print_hits(result.hits)
    print("\n💬 回答：\n")
    print(result.text)
    print(f"\n（{result.provider}/{result.model or '-'}，耗时 {result.elapsed_ms:.0f}ms）")
    if trace is not None:
        print("\n⏱️  链路耗时：")
        for step in trace.steps:
            print(f"    {step['step']:<16} {step['ms']:>7.1f}ms  {step['detail']}")
        print(f"    合计 {trace.total_ms}ms")
    return 0


def cmd_interactive(args) -> int:
    print("🔍 RAG 问答已启动（输入 exit 退出，clear 清空上下文）")
    history: list[dict[str, str]] = []
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if question.lower() in {"exit", "quit"}:
            print("再见！")
            return 0
        if question.lower() == "clear":
            history.clear()
            print("上下文已清空。")
            continue
        if not question:
            continue
        answer_once(question, args, history)
        history.append({"role": "user", "content": question})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="通用 RAG 文档问答（CLI）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="*", help="要问的问题；不填则进入交互模式")
    parser.add_argument("--rebuild", action="store_true", help="按 config.DOC_PATH 重建索引")
    parser.add_argument("--ingest", metavar="路径", help="入库指定的文件/目录（支持逗号分隔多个）")
    parser.add_argument("--append", action="store_true", help="配合 --ingest：追加而不是清空重建")
    parser.add_argument("--stats", action="store_true", help="查看向量库状态")
    parser.add_argument("--prune", action="store_true", help="清理向量库残留段目录")
    parser.add_argument("--doctor", action="store_true", help="环境自检")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="召回片段数")
    parser.add_argument("--mode", choices=("vector", "keyword", "hybrid"),
                        default=config.RETRIEVAL_MODE, help="检索模式")
    parser.add_argument("--provider", default=None, help="覆盖 LLM_PROVIDER")
    parser.add_argument("--trace", action="store_true", help="打印检索链路耗时")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging_config.setup_logging("WARNING" if not args.trace else "INFO")

    if args.doctor:
        return cmd_doctor()
    if args.stats:
        return cmd_stats()
    if args.prune:
        return cmd_prune()
    if args.ingest:
        return cmd_ingest(args.ingest, args.append)
    if args.rebuild:
        import ingest

        report = ingest.ingest_sources(reset=True, doc_type=config.DOC_TYPE)
        print(report.summary())
        return 0 if report.chunks else 1

    question = " ".join(args.question).strip()
    if question:
        return answer_once(question, args)
    return cmd_interactive(args)


if __name__ == "__main__":
    sys.exit(main())
