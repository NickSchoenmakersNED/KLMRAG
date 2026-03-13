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

EMBEDDING_CONFIGS = [
    {
        "label": "qwen3-0.6b | chunk=500 | overlap=50",
        "model": "text-embedding-qwen3-embedding-0.6b",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "k": 3,
    },
    {
        "label": "qwen3-0.6b | chunk=300 | overlap=50",
        "model": "text-embedding-qwen3-embedding-0.6b",
        "chunk_size": 300,
        "chunk_overlap": 50,
        "k": 3,
    },
    {
        "label": "qwen3-0.6b | chunk=500 | overlap=100",
        "model": "text-embedding-qwen3-embedding-0.6b",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "k": 3,
    },
    # Add more configs to compare — swap model name when you have a second model
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


def chunk_contains_keyword(chunk_text: str, keywords: list[str]) -> bool:
    text = chunk_text.lower()
    return any(kw.lower() in text for kw in keywords)


def run_retrieval_test(config: dict, docs: list) -> list[RetrievalResult]:
    embeddings = OpenAIEmbeddings(
        check_embedding_ctx_length=False,
        model=config["model"],
        api_key="not-needed",
        base_url=BASE_URL,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)

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
        # For out_of_scope entries, success means the keywords were NOT retrieved.
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


# ── Reporting ─────────────────────────────────────────────────────────────────

def _type_breakdown(results: list[RetrievalResult]) -> dict[str, tuple[int, int]]:
    """Returns {question_type: (successes, total)} sorted by type name."""
    by_type: dict[str, list[RetrievalResult]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)
    return {
        qtype: (sum(1 for r in group if r.success), len(group))
        for qtype, group in sorted(by_type.items())
    }


def print_report(config_label: str, results: list[RetrievalResult]):
    total    = len(results)
    successes = sum(1 for r in results if r.success)

    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config_label}")
    print(f"Overall success@k: {successes}/{total}  ({100 * successes / total:.0f}%)")
    print(f"  (out_of_scope entries: success = keyword NOT retrieved)")
    print(f"{'=' * 70}")

    for r in results:
        if r.is_out_of_scope:
            status   = "✓ CLEAN" if r.success else "✗ LEAK"
            rank_str = "correctly absent" if r.success else f"leaked at rank {r.first_hit_rank}"
        else:
            status   = "✓ HIT" if r.success else "✗ MISS"
            rank_str = f"rank {r.first_hit_rank}" if r.first_hit_rank else "not found"

        print(f"\n[{status}] {r.question_id} ({rank_str}) | top_score={r.top_score:.4f} | type={r.question_type}")
        print(f"  Q: {r.question[:80]}")
        print(f"  Keywords: {r.expected_keywords}")
        for i, preview in enumerate(r.retrieved_previews, 1):
            marker = "  >>>" if (r.first_hit_rank == i) else "     "
            print(f"{marker} [{i}] {preview}")
        if r.notes:
            print(f"  note: {r.notes}")

    # ── Per-type breakdown ────────────────────────────────────────────────────
    breakdown = _type_breakdown(results)
    col_w = max(len(t) for t in breakdown) + 2

    print(f"\n{'─' * 70}")
    print("BREAKDOWN BY TYPE")
    print(f"{'─' * 70}")
    for qtype, (s, t) in breakdown.items():
        pct = 100 * s / t
        bar = "█" * s + "░" * (t - s)
        print(f"  {qtype:<{col_w}} {s}/{t}  ({pct:.0f}%)  {bar}")
    print(f"{'─' * 70}")
    print(f"  {'TOTAL':<{col_w}} {successes}/{total}  ({100 * successes / total:.0f}%)")


def save_results(all_results: dict):
    """Save raw results to JSON for later analysis."""
    serializable = {}
    for label, results in all_results.items():
        serializable[label] = [
            {
                "id": r.question_id,
                "type": r.question_type,
                "is_out_of_scope": r.is_out_of_scope,
                "hit": r.hit,
                "success": r.success,
                "first_hit_rank": r.first_hit_rank,
                "top_score": r.top_score,
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

    all_results: dict[str, list[RetrievalResult]] = {}

    for config in EMBEDDING_CONFIGS:
        print(f"\nRunning config: {config['label']}")
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

    # Header
    label_w = max(len(lbl) for lbl in all_results) + 2
    header  = f"  {'TYPE':<{col_w}}" + "".join(f"  {lbl:<{label_w}}" for lbl in all_results)
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for qtype in all_types + ["TOTAL"]:
        row = f"  {qtype:<{col_w}}"
        for label, results in all_results.items():
            if qtype == "TOTAL":
                s = sum(1 for r in results if r.success)
                t = len(results)
            else:
                group = [r for r in results if r.question_type == qtype]
                s = sum(1 for r in group if r.success)
                t = len(group)
            row += f"  {s}/{t} ({100*s/t:.0f}%){'':<{label_w - 10}}"
        print(row)