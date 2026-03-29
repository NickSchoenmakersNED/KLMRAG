import sys
import time
import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

sys.path.append(str(Path(__file__).parent.parent))

from main import retriever, format_docs_with_sources, finalize_response_with_sources, llm as judge_llm
from eval_dataset import RESPONSE_EVAL_SET


# =============================================================================
# CONFIG — EDIT THESE
# =============================================================================

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
NUM_RUNS = 3

MODELS = [
    {"name": "Qwen2.5-14B-Instruct-1M",       "model_id": "Qwen/Qwen2.5-14B-Instruct-1M"},
    {"name": "Galactic-Qwen-14B-Exp2",         "model_id": "prithivMLmods/Galactic-Qwen-14B-Exp2"},
    {"name": "T3Q-qwen2.5-14b-v1.0-e3",        "model_id": "JungZoona/T3Q-qwen2.5-14b-v1.0-e3"},
    {"name": "Q2.5-Veltha-14B",                 "model_id": "djuna/Q2.5-Veltha-14B"},
    {"name": "Qwen2.5-14B-Instruct-abliterated","model_id": "huihui-ai/Qwen2.5-14B-Instruct-abliterated-v2"},
    {"name": "Rombos-LLM-V2.6-Qwen-14b",       "model_id": "rombodawg/Rombos-LLM-V2.6-Qwen-14b"},
    {"name": "ZYH-LLM-Qwen2.5-14B-V3",         "model_id": "YOYO-AI/ZYH-LLM-Qwen2.5-14B-V3"},
]

# ---------------------------------------------------------------------------
# IMPORTANT: Replace this with your actual RAG generation prompt from main.py.
# It must accept {context}, {history}, and {question} as template variables.
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """Je bent een klantenservice-assistent voor een luchtvaartmaatschappij.
Beantwoord de vraag uitsluitend op basis van de onderstaande context.
Als het antwoord niet in de context staat, zeg dan eerlijk dat je het niet weet.

Context:
{context}

Gespreksgeschiedenis:
{history}

Vraag: {question}

Antwoord:"""


# =============================================================================
# JUDGE SETUP (uses your existing Gemma model from main.py)
# =============================================================================

EVALUATOR_PROMPT = """Je bent een strenge, objectieve beoordelaar van AI-systemen.
Beoordeel het onderstaande AI-antwoord op basis van de vraag, de verwachte waarheid (ground truth), en de verstrekte context.

Geef je beoordeling als een JSON-object met exact deze sleutels:
- "faithfulness" (boolean): Is de stelling in het "AI Antwoord" gebaseerd op de "Context"? (False als de AI verzint/hallucineert).
- "correctness" (boolean): Komt de inhoud van het "AI Antwoord" overeen met de "Verwachte Waarheid"?
- "citation_accuracy" (boolean): Zijn de inline bronverwijzingen correct geplaatst bij de juiste informatie?
- "reasoning" (string): Korte uitleg (max 3 zinnen) van je beoordeling.

Vraag: {question}
Verwachte Waarheid: {ground_truth}

Context:
{context}

---
AI Antwoord:
{answer}
"""

eval_prompt = ChatPromptTemplate.from_template(EVALUATOR_PROMPT)
eval_chain = eval_prompt | judge_llm | JsonOutputParser()


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def build_generation_chain(model_config: dict):
    llm = ChatOpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key=LMSTUDIO_API_KEY,
        model=model_config["model_id"],
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return prompt | llm | StrOutputParser()


def verify_model_connection(chain) -> bool:
    try:
        response = chain.invoke({
            "context": "Test context.",
            "history": "",
            "question": "Zeg alleen 'OK' als je werkt."
        })
        return bool(response and len(response.strip()) > 0)
    except Exception as e:
        print(f"  Connection failed: {e}")
        return False


def run_single_evaluation(entry: dict, generation_chain, docs, context: str) -> dict:
    start_time = time.perf_counter()
    raw_answer = generation_chain.invoke({
        "context": context,
        "history": "",
        "question": entry["question"],
    })
    end_time = time.perf_counter()

    latency = end_time - start_time
    final_answer = finalize_response_with_sources(raw_answer, docs)
    output_chars = len(final_answer)

    try:
        evaluation = eval_chain.invoke({
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
            "context": context,
            "answer": final_answer,
        })
    except Exception as e:
        evaluation = {
            "faithfulness": None,
            "correctness": None,
            "citation_accuracy": None,
            "reasoning": f"Judge parse error: {e}",
        }

    return {
        "id": entry["id"],
        "type": entry["type"],
        "latency_seconds": round(latency, 3),
        "output_chars": output_chars,
        "ai_response": final_answer,
        "evaluation": evaluation,
    }


def run_benchmark_for_model(model_config: dict, num_runs: int) -> dict:
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_config['name']}")
    print(f"Model ID:     {model_config['model_id']}")
    print(f"Runs:         {num_runs}")
    print(f"{'='*60}")

    generation_chain = build_generation_chain(model_config)

    print("  Verifying connection to LM Studio...")
    if not verify_model_connection(generation_chain):
        print("  SKIPPED — could not connect. Is this model loaded in LM Studio?")
        return {"skipped": True, "reason": "connection_failed"}

    print("  Connection OK. Starting evaluation runs.\n")

    prefetched = []
    for entry in RESPONSE_EVAL_SET:
        docs = retriever.invoke(entry["question"])
        context = format_docs_with_sources(docs)
        prefetched.append({"entry": entry, "docs": docs, "context": context})

    all_runs = []
    for run_num in range(1, num_runs + 1):
        print(f"  --- Run {run_num}/{num_runs} ---")
        run_results = []

        for item in prefetched:
            result = run_single_evaluation(
                item["entry"], generation_chain, item["docs"], item["context"]
            )
            status_f = format_bool(result["evaluation"].get("faithfulness"))
            status_c = format_bool(result["evaluation"].get("correctness"))
            print(f"    [{result['id']}] F:{status_f} C:{status_c} | {result['latency_seconds']}s")
            run_results.append(result)

        all_runs.append({"run_number": run_num, "results": run_results})

    aggregate = compute_aggregate(all_runs)
    return {"skipped": False, "runs": all_runs, "aggregate": aggregate}


def format_bool(val) -> str:
    if val is True:
        return "✅"
    if val is False:
        return "❌"
    return "⚠️"


def safe_mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def safe_stdev(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.stdev(clean), 4) if len(clean) >= 2 else 0.0


def compute_aggregate(all_runs: list[dict]) -> dict:
    per_run_faithfulness = []
    per_run_correctness = []
    per_run_citation = []
    per_run_latency = []

    for run in all_runs:
        results = run["results"]
        valid = [r for r in results if r["evaluation"].get("faithfulness") is not None]

        if valid:
            per_run_faithfulness.append(
                sum(1 for r in valid if r["evaluation"]["faithfulness"]) / len(valid)
            )
            per_run_correctness.append(
                sum(1 for r in valid if r["evaluation"]["correctness"]) / len(valid)
            )
            per_run_citation.append(
                sum(1 for r in valid if r["evaluation"].get("citation_accuracy")) / len(valid)
            )

        latencies = [r["latency_seconds"] for r in results]
        per_run_latency.append(statistics.mean(latencies))

    return {
        "faithfulness":     {"mean": safe_mean(per_run_faithfulness), "std": safe_stdev(per_run_faithfulness)},
        "correctness":      {"mean": safe_mean(per_run_correctness),  "std": safe_stdev(per_run_correctness)},
        "citation_accuracy":{"mean": safe_mean(per_run_citation),     "std": safe_stdev(per_run_citation)},
        "avg_latency_sec":  {"mean": safe_mean(per_run_latency),      "std": safe_stdev(per_run_latency)},
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(all_model_results: dict):
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    header = f"{'Model':<40} {'Faith':>8} {'Correct':>8} {'Citation':>8} {'Latency':>10}"
    print(header)
    print("-" * 80)

    for model_name, data in all_model_results.items():
        if data.get("skipped"):
            print(f"{model_name:<40} {'SKIPPED':>8}")
            continue

        agg = data["aggregate"]
        f_str = format_metric(agg["faithfulness"])
        c_str = format_metric(agg["correctness"])
        ci_str = format_metric(agg["citation_accuracy"])
        l_str = format_latency(agg["avg_latency_sec"])
        print(f"{model_name:<40} {f_str:>8} {c_str:>8} {ci_str:>8} {l_str:>10}")

    print(f"{'='*80}")


def format_metric(m: dict) -> str:
    if m["mean"] is None:
        return "N/A"
    pct = m["mean"] * 100
    std = m["std"] * 100 if m["std"] else 0
    return f"{pct:.0f}±{std:.0f}%"


def format_latency(m: dict) -> str:
    if m["mean"] is None:
        return "N/A"
    return f"{m['mean']:.1f}±{m['std']:.1f}s"


def save_results(all_model_results: dict, output_path: str):
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_runs": NUM_RUNS,
            "eval_set_size": len(RESPONSE_EVAL_SET),
            "judge_model": "gemma (from main.py)",
            "lmstudio_base_url": LMSTUDIO_BASE_URL,
        },
        "models": {},
    }

    for model_name, data in all_model_results.items():
        if data.get("skipped"):
            report["models"][model_name] = {"skipped": True, "reason": data.get("reason")}
        else:
            report["models"][model_name] = {
                "aggregate": data["aggregate"],
                "runs": data["runs"],
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RAG LLM Benchmark Harness")
    parser.add_argument("--model", type=str, help="Run only this model (by name from MODELS list)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help=f"Runs per model (default: {NUM_RUNS})")
    parser.add_argument("--no-pause", action="store_true", help="Skip pause between models (if you have auto-loading)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    num_runs = args.runs

    if args.model:
        targets = [m for m in MODELS if m["name"] == args.model]
        if not targets:
            available = [m["name"] for m in MODELS]
            print(f"Model '{args.model}' not found. Available: {available}")
            sys.exit(1)
    else:
        targets = MODELS

    all_model_results = {}

    for i, model_config in enumerate(targets):
        if i > 0 and not args.no_pause:
            print(f"\n{'*'*60}")
            print(f"NEXT MODEL: {model_config['name']}")
            print(f"Load this model in LM Studio, then press Enter.")
            print(f"{'*'*60}")
            input(">> Press Enter when ready... ")

        result = run_benchmark_for_model(model_config, num_runs)
        all_model_results[model_config["name"]] = result

        save_results(all_model_results, args.output)

    print_summary(all_model_results)
    save_results(all_model_results, args.output)


if __name__ == "__main__":
    main()