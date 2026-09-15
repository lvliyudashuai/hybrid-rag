"""离线评估：用一份带标准答案的小数据集，量化「检索到底好不好」。

为什么必须有这个？改了分块大小、换了模型、调了 alpha…… 如果只靠"感觉好像
好一点"，RAG 是没法优化的。有了评估集，每次改动都能看到具体数字涨了还是跌了。

数据集是 JSONL，一行一个问题：
    {"question": "猪八戒的武器有多重",
     "expected_sections": ["第八九回"],      # 命中这些章节算对（子串匹配）
     "expected_keywords": ["八百斤"],         # 关键词是否出现在召回内容里
     "answerable": true}                      # false 表示文档里没有答案

用法：
    python evaluate.py                          # 默认数据集 + 三种模式对比
    python evaluate.py --dataset eval/qa_sample.jsonl --k 5
    python evaluate.py --mode hybrid --show-failures
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
import logging_config

DEFAULT_DATASET = config.BASE_DIR / "eval" / "qa_sample.jsonl"
MODES = ("vector", "keyword", "hybrid")


@dataclass
class Case:
    question: str
    expected_sections: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    answerable: bool = True


@dataclass
class CaseResult:
    case: Case
    sections: list[str]
    context: str
    hit: bool
    rank: int | None
    keyword_recall: float
    latency_ms: float


@dataclass
class ModeReport:
    mode: str
    k: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for item in self.results if item.hit) / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        total = sum(1.0 / item.rank for item in self.results if item.rank)
        return total / len(self.results)

    @property
    def keyword_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(item.keyword_recall for item in self.results) / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(item.latency_ms for item in self.results) / len(self.results)

    def failures(self) -> list[CaseResult]:
        return [item for item in self.results if not item.hit]

    def as_row(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "k": self.k,
            "cases": len(self.results),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "keyword_recall": round(self.keyword_recall, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


def load_dataset(path: Path) -> list[Case]:
    """读取 JSONL 数据集，跳过空行和 # 注释行。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到评估集：{path}")
    cases: list[Case] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {number} 行不是合法 JSON：{exc}") from exc
        cases.append(
            Case(
                question=payload["question"],
                expected_sections=payload.get("expected_sections", []),
                expected_keywords=payload.get("expected_keywords", []),
                answerable=payload.get("answerable", True),
            )
        )
    return cases


def evaluate_case(case: Case, hits, k: int) -> CaseResult:
    """判定单个问题：命中章节算对，顺带算关键词覆盖率。"""
    sections = [hit.section for hit in hits[:k]]
    context = "\n".join(hit.text for hit in hits[:k])

    rank: int | None = None
    for index, section in enumerate(sections, 1):
        if any(expected in section for expected in case.expected_sections):
            rank = index
            break
    # 没有标准章节时（例如"文档里没有答案"的反例），不做命中判定
    hit = rank is not None if case.expected_sections else True

    if case.expected_keywords:
        found = sum(1 for word in case.expected_keywords if word in context)
        keyword_recall = found / len(case.expected_keywords)
    else:
        keyword_recall = 1.0
    return CaseResult(case=case, sections=sections, context=context, hit=hit,
                      rank=rank, keyword_recall=keyword_recall, latency_ms=0.0)


def run_mode(cases: list[Case], mode: str, k: int, **search_kwargs) -> ModeReport:
    """在给定模式下跑完整数据集。"""
    from retrieval import search

    report = ModeReport(mode=mode, k=k)
    for case in cases:
        started = time.perf_counter()
        hits = search(case.question, k=k, mode=mode, **search_kwargs)
        elapsed = (time.perf_counter() - started) * 1000
        result = evaluate_case(case, hits, k)
        result.latency_ms = elapsed
        report.results.append(result)
    return report


def warmup() -> None:
    """先跑一次丢弃结果的检索。

    第一次调用要把 embedding 模型加载进显存（几秒），不预热的话这笔开销会
    全算到第一个模式头上，让耗时对比失真。
    """
    from retrieval import search

    with contextlib.suppress(Exception):
        search("预热", 1)


def render_table(reports: list[ModeReport]) -> str:
    header = f"{'模式':<10}{'Hit@k':>9}{'MRR':>9}{'关键词覆盖':>12}{'平均耗时':>12}"
    lines = [header, "-" * len(header)]
    for report in reports:
        lines.append(
            f"{report.mode:<10}{report.hit_rate:>9.1%}{report.mrr:>9.3f}"
            f"{report.keyword_recall:>12.1%}{report.avg_latency_ms:>10.0f}ms"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量离线评估")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL 评估集路径")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="每次召回的片段数")
    parser.add_argument("--mode", choices=MODES + ("all",), default="all",
                        help="检索模式；all 表示三种都跑一遍做对比")
    parser.add_argument("--show-failures", action="store_true", help="打印没命中的问题")
    parser.add_argument("--json", dest="json_out", default="", help="把结果写成 JSON 文件")
    args = parser.parse_args(argv)

    logging_config.setup_logging("WARNING")
    cases = load_dataset(Path(args.dataset))
    if not cases:
        print("评估集是空的。")
        return 1

    print(f"📋 评估集：{args.dataset}（{len(cases)} 个问题，k={args.k}）\n")
    warmup()
    modes = MODES if args.mode == "all" else (args.mode,)
    reports = [run_mode(cases, mode, args.k) for mode in modes]
    print(render_table(reports))

    if args.show_failures:
        for report in reports:
            failures = report.failures()
            if not failures:
                continue
            print(f"\n❌ {report.mode} 未命中 {len(failures)} 个：")
            for item in failures:
                print(f"   · {item.case.question}")
                print(f"     期望章节：{item.case.expected_sections}")
                print(f"     实际召回：{item.sections[:3]}")

    if args.json_out:
        payload = {
            "dataset": args.dataset,
            "k": args.k,
            "reports": [report.as_row() for report in reports],
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
