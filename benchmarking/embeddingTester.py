"""
embedding_tester.py

Tests retrieval ONLY — no LLM involved.
Measures whether the right chunks come back for each question.

Metrics:
  - hit@k        : did ANY retrieved chunk contain an expected keyword?
  - rank         : at what position did the first hit appear?
  - similarity   : raw cosine score of the top result

Special handling for out_of_scope entries: a retrieval HIT is counted as a
FAILURE (the system should not have returned in-scope content). Their success
flag is inverted so they don't silently distort global and per-type recall.

Each config is run NUM_RUNS times. Per-question metrics are averaged across
runs before reporting. success_rate is a float in [0.0, 1.0]; a question is
considered successful if success_rate >= 0.5.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
import re
import yaml

from eval_dataset import EVAL_SET


# ── Config ────────────────────────────────────────────────────────────────────

NUM_RUNS = 3  # Number of times each config is run before averaging

EMBEDDING_CONFIGS = [
    # {
    #     "label": "qwen3-0.6b | chunk=500 | overlap=50 | k=3",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 500,
    #     "chunk_overlap": 50,
    #     "k": 3,
    # },
    # {
    #     "label": "qwen3-0.6b | chunk=600 | overlap=50 | k=3",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 600,
    #     "chunk_overlap": 50,
    #     "k": 3,
    # },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     4.00/5  (80%)  ████░
    #   happy_path    10.00/12  (83%)  ██████████░░
    #   misleading    3.00/4  (75%)  ███░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         21.00/25  (84%)
    {
        "label": "qwen3-0.6b | chunk=600 | overlap=50 | k=4",
        "model": "text-embedding-qwen3-embedding-0.6b",
        "chunk_size": 600,
        "chunk_overlap": 50,
        "k": 4,
    },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    11.00/12  (92%)  ███████████░
    #   misleading    3.00/4  (75%)  ███░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         23.00/25  (92%)
    # {
    #     "label": "qwen3-0.6b | chunk=600 | overlap=50 | k=4",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 600,
    #     "chunk_overlap": 75,
    #     "k": 4,
    # },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    11.00/12  (92%)  ███████████░
    #   misleading    2.00/4  (50%)  ██░░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         22.00/25  (88%)
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    11.00/12  (92%)  ███████████░
    #   misleading    2.00/4  (50%)  ██░░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         22.00/25  (88%)
    # {
    #     "label": "qwen3-0.6b | chunk=700 | overlap=50 | k=4",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 700,
    #     "chunk_overlap": 50,
    #     "k": 4,
    # },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    12.00/12  (100%)  ████████████
    #   misleading    2.00/4  (50%)  ██░░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         23.00/25  (92%)
    # {
    #     "label": "qwen3-0.6b | chunk=500 | overlap=50 | k=5",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 500,
    #     "chunk_overlap": 50,
    #     "k": 5,
    # },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    11.00/12  (92%)  ███████████░
    #   misleading    2.00/4  (50%)  ██░░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         22.00/25  (88%)
    # {
    #     "label": "qwen3-0.6b | chunk=600 | overlap=50 | k=5",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 600,
    #     "chunk_overlap": 50,
    #     "k": 5,
    # },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    11.00/12  (92%)  ███████████░
    #   misleading    3.00/4  (75%)  ███░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         23.00/25  (92%)
    # {
    #     "label": "qwen3-0.6b | chunk=600 | overlap=50 | k=5",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 600,
    #     "chunk_overlap": 75,
    #     "k": 5,
    # },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    11.00/12  (92%)  ███████████░
    #   misleading    2.00/4  (50%)  ██░░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    # {
    #     "label": "qwen3-0.6b | chunk=600 | overlap=50 | k=5",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 700,
    #     "chunk_overlap": 50,
    #     "k": 5,
    # },
    #     ──────────────────────────────────────────────────────────────────────
    #   edge_case     5.00/5  (100%)  █████
    #   happy_path    12.00/12  (100%)  ████████████
    #   misleading    2.00/4  (50%)  ██░░
    #   noisy_input   4.00/4  (100%)  ████
    # ──────────────────────────────────────────────────────────────────────
    #   TOTAL         23.00/25  (92%)
    # {
    #     "label": "qwen3-0.6b | chunk=400 | overlap=50 | k=3",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 400,
    #     "chunk_overlap": 50,
    #     "k": 3,
    # },
    # {
    #     "label": "qwen3-0.6b | chunk=250 | overlap=25 | k=6",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 250,
    #     "chunk_overlap": 25,
    #     "k": 6,
    # },
    # {
    #     "label": "qwen3-0.6b | chunk=100 | overlap=10 | k=10",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 100,
    #     "chunk_overlap": 10,
    #     "k": 10,
    # },
    # {
    #     "label": "qwen3-0.6b | chunk=1000 | overlap=100 | k=2",
    #     "model": "text-embedding-qwen3-embedding-0.6b",
    #     "chunk_size": 1000,
    #     "chunk_overlap": 100,
    #     "k": 2,
    # },
]

BASE_URL = "http://localhost:1234/v1"

PDF_PATH = r"data\raw\cellar_439cd3a7-fd3c-4da7-8bf4-b0f60600c1d6.0004.02_DOC_1.pdf"
MD_DIR   = "data/processed"


# ── Document loading ──────────────────────────────────────────────────────────

def load_markdown_with_metadata(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    metadata = {}
    body = content
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if match:
        try:
            metadata = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            pass
        body = content[match.end():]
    return Document(page_content=body, metadata=metadata)


def load_all_docs():
    pages = PyPDFLoader(PDF_PATH).load()
    md_docs = [load_markdown_with_metadata(p) for p in Path(MD_DIR).glob("*.md")]
    return pages + md_docs


# ── Core test logic ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    question_id: str
    question: str
    question_type: str             # happy_path | misleading | noisy_input | edge_case | out_of_scope
    is_out_of_scope: bool          # True → success means NO keyword found
    hit: bool                      # raw: did any chunk contain an expected keyword?
    success: bool                  # hit XOR is_out_of_scope — the metric that actually matters
    first_hit_rank: Optional[int]  # 1-indexed rank of first matching chunk (None = miss)
    top_score: float               # cosine similarity of rank-1 chunk
    retrieved_previews: list[str]  # first 120 chars of each retrieved chunk
    expected_keywords: list[str]
    notes: str = ""


@dataclass
class AveragedResult:
    """Per-question metrics averaged across NUM_RUNS runs."""
    question_id: str
    question: str
    question_type: str
    is_out_of_scope: bool
    expected_keywords: list[str]
    notes: str

    success_rate: float            # fraction of runs where success=True  (0.0–1.0)
    hit_rate: float                # fraction of runs where hit=True
    avg_top_score: float           # mean cosine similarity of rank-1 chunk
    avg_first_hit_rank: float      # mean rank of first hit (None runs use k+1 as sentinel)

    # Derived for reporting
    success: bool = field(init=False)          # success_rate >= 0.5
    first_hit_rank: Optional[float] = field(init=False)  # None if avg rank == sentinel

    # Representative previews from the last run
    retrieved_previews: list[str] = field(default_factory=list)

    _k_sentinel: int = field(repr=False, default=0)  # k+1 used for None ranks

    def __post_init__(self):
        self.success = self.success_rate >= 0.5
        # If the average rank equals the sentinel, nothing was ever found
        self.first_hit_rank = (
            None if self.avg_first_hit_rank >= self._k_sentinel
            else self.avg_first_hit_rank
        )


def chunk_contains_keyword(chunk_text: str, keywords: list[str]) -> bool:
    text = chunk_text.lower()
    return any(kw.lower() in text for kw in keywords)


def run_single(config: dict, docs: list, chunks: list) -> list[RetrievalResult]:
    """One retrieval pass over the eval set. Chunks are pre-split."""
    embeddings = OpenAIEmbeddings(
        check_embedding_ctx_length=False,
        model=config["model"],
        api_key="not-needed",
        base_url=BASE_URL,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    k = config["k"]

    results = []
    for entry in EVAL_SET:
        is_out_of_scope = entry.get("type") == "out_of_scope"

        hits_with_scores = vectorstore.similarity_search_with_score(entry["question"], k=k)

        first_hit_rank = None
        for rank, (doc, _score) in enumerate(hits_with_scores, start=1):
            if chunk_contains_keyword(doc.page_content, entry["expected_keywords"]):
                first_hit_rank = rank
                break

        hit = first_hit_rank is not None
        success = (not hit) if is_out_of_scope else hit

        top_score = hits_with_scores[0][1] if hits_with_scores else 0.0
        previews = [doc.page_content[:120].replace("\n", " ") for doc, _ in hits_with_scores]

        results.append(RetrievalResult(
            question_id=entry["id"],
            question=entry["question"],
            question_type=entry.get("type", "untagged"),
            is_out_of_scope=is_out_of_scope,
            hit=hit,
            success=success,
            first_hit_rank=first_hit_rank,
            top_score=top_score,
            retrieved_previews=previews,
            expected_keywords=entry["expected_keywords"],
            notes=entry.get("notes", ""),
        ))

    return results


def average_runs(
    all_runs: list[list[RetrievalResult]],
    k: int,
) -> list[AveragedResult]:
    """Collapse NUM_RUNS result lists into one averaged list, ordered by question_id."""
    sentinel = k + 1  # rank assigned when a question was never hit in a run

    # Index runs by question_id
    by_id: dict[str, list[RetrievalResult]] = defaultdict(list)
    for run in all_runs:
        for r in run:
            by_id[r.question_id].append(r)

    averaged = []
    for qid, runs in by_id.items():
        ref = runs[0]  # use first run for stable fields
        n = len(runs)

        success_rate      = sum(r.success    for r in runs) / n
        hit_rate          = sum(r.hit        for r in runs) / n
        avg_top_score     = sum(r.top_score  for r in runs) / n
        avg_first_hit_rank = sum(
            (r.first_hit_rank if r.first_hit_rank is not None else sentinel)
            for r in runs
        ) / n

        # Previews from the last run (most recent context for debugging)
        previews = runs[-1].retrieved_previews

        averaged.append(AveragedResult(
            question_id=qid,
            question=ref.question,
            question_type=ref.question_type,
            is_out_of_scope=ref.is_out_of_scope,
            expected_keywords=ref.expected_keywords,
            notes=ref.notes,
            success_rate=success_rate,
            hit_rate=hit_rate,
            avg_top_score=avg_top_score,
            avg_first_hit_rank=avg_first_hit_rank,
            retrieved_previews=previews,
            _k_sentinel=sentinel,
        ))

    # Preserve original eval set ordering
    order = {entry["id"]: i for i, entry in enumerate(EVAL_SET)}
    averaged.sort(key=lambda r: order.get(r.question_id, 9999))
    return averaged


def run_retrieval_test(config: dict, docs: list) -> list[AveragedResult]:
    """Split once, embed NUM_RUNS times, return averaged results."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

    all_runs: list[list[RetrievalResult]] = []
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"  run {run_idx}/{NUM_RUNS}...", end=" ", flush=True)
        run_results = run_single(config, docs, chunks)
        all_runs.append(run_results)
        successes = sum(1 for r in run_results if r.success)
        print(f"success {successes}/{len(run_results)}")

    return average_runs(all_runs, k=config["k"])


# ── Reporting ─────────────────────────────────────────────────────────────────

def _type_breakdown(results: list[AveragedResult]) -> dict[str, tuple[float, int]]:
    """Returns {question_type: (avg_successes, total)} sorted by type name."""
    by_type: dict[str, list[AveragedResult]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)
    return {
        qtype: (sum(r.success_rate for r in group), len(group))
        for qtype, group in sorted(by_type.items())
    }


def print_report(config_label: str, results: list[AveragedResult]):
    total     = len(results)
    # Sum of success_rates gives effective successes across runs
    eff_succ  = sum(r.success_rate for r in results)

    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config_label}  (averaged over {NUM_RUNS} runs)")
    print(f"Overall success@k: {eff_succ:.2f}/{total}  ({100 * eff_succ / total:.0f}%)")
    print(f"  (out_of_scope entries: success = keyword NOT retrieved)")
    print(f"{'=' * 70}")

    for r in results:
        if r.is_out_of_scope:
            status   = "✓ CLEAN" if r.success else "✗ LEAK"
            rank_str = (
                "correctly absent"
                if r.first_hit_rank is None
                else f"leaked ~rank {r.first_hit_rank:.1f}"
            )
        else:
            status   = "✓ HIT" if r.success else "✗ MISS"
            rank_str = (
                f"~rank {r.first_hit_rank:.1f}" if r.first_hit_rank is not None
                else "not found"
            )

        print(
            f"\n[{status}] {r.question_id} ({rank_str}) | "
            f"top_score={r.avg_top_score:.4f} | "
            f"success_rate={r.success_rate:.2f} | "
            f"type={r.question_type}"
        )
        print(f"  Q: {r.question[:80]}")
        print(f"  Keywords: {r.expected_keywords}")

        # Mark the average hit rank position (rounded) in the preview list
        avg_rank_int = (
            round(r.first_hit_rank) if r.first_hit_rank is not None else None
        )
        for i, preview in enumerate(r.retrieved_previews, 1):
            marker = "  >>>" if avg_rank_int == i else "     "
            print(f"{marker} [{i}] {preview}")

        if r.notes:
            print(f"  note: {r.notes}")

    # ── Per-type breakdown ────────────────────────────────────────────────────
    breakdown = _type_breakdown(results)
    col_w = max(len(t) for t in breakdown) + 2

    print(f"\n{'─' * 70}")
    print("BREAKDOWN BY TYPE")
    print(f"{'─' * 70}")
    for qtype, (eff_s, t) in breakdown.items():
        pct = 100 * eff_s / t
        filled = round(eff_s)
        bar = "█" * filled + "░" * (t - filled)
        print(f"  {qtype:<{col_w}} {eff_s:.2f}/{t}  ({pct:.0f}%)  {bar}")
    print(f"{'─' * 70}")
    print(f"  {'TOTAL':<{col_w}} {eff_succ:.2f}/{total}  ({100 * eff_succ / total:.0f}%)")


def save_results(all_results: dict):
    """Save averaged results to JSON for later analysis."""
    serializable = {}
    for label, results in all_results.items():
        serializable[label] = [
            {
                "id": r.question_id,
                "type": r.question_type,
                "is_out_of_scope": r.is_out_of_scope,
                "success": r.success,
                "success_rate": round(r.success_rate, 4),
                "hit_rate": round(r.hit_rate, 4),
                "avg_first_hit_rank": (
                    round(r.avg_first_hit_rank, 2)
                    if r.first_hit_rank is not None else None
                ),
                "avg_top_score": round(r.avg_top_score, 6),
                "previews": r.retrieved_previews,
            }
            for r in results
        ]
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print("\nResults saved to eval_results.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_all_docs()
    print(f"Loaded {len(docs)} documents")

    all_results: dict[str, list[AveragedResult]] = {}

    for config in EMBEDDING_CONFIGS:
        print(f"\nRunning config: {config['label']}  ({NUM_RUNS} runs)")
        results = run_retrieval_test(config, docs)
        print_report(config["label"], results)
        all_results[config["label"]] = results

    save_results(all_results)

    # ── Cross-config summary ──────────────────────────────────────────────────
    all_types = sorted({r.question_type for results in all_results.values() for r in results})
    col_w     = max(len(t) for t in all_types) + 2

    print(f"\n{'=' * 70}")
    print("CROSS-CONFIG SUMMARY")
    print(f"{'=' * 70}")

    label_w = max(len(lbl) for lbl in all_results) + 2
    header  = f"  {'TYPE':<{col_w}}" + "".join(f"  {lbl:<{label_w}}" for lbl in all_results)
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for qtype in all_types + ["TOTAL"]:
        row = f"  {qtype:<{col_w}}"
        for label, results in all_results.items():
            if qtype == "TOTAL":
                eff_s = sum(r.success_rate for r in results)
                t = len(results)
            else:
                group = [r for r in results if r.question_type == qtype]
                eff_s = sum(r.success_rate for r in group)
                t = len(group)
            row += f"  {eff_s:.2f}/{t} ({100*eff_s/t:.0f}%){'':<{label_w - 14}}"
        print(row)