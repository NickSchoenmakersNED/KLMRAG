import time
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader


# ── Benchmarking infrastructure ───────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    label: str
    elapsed_ms: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    chunks_retrieved: Optional[int] = None
    answer_chars: Optional[int] = None


@dataclass
class BenchmarkSession:
    results: list[BenchmarkResult] = field(default_factory=list)

    def record(self, result: BenchmarkResult) -> None:
        self.results.append(result)
        self._print(result)

    @staticmethod
    def _print(r: BenchmarkResult) -> None:
        extras = []
        if r.tokens_in  is not None: extras.append(f"tokens_in={r.tokens_in}")
        if r.tokens_out is not None: extras.append(f"tokens_out={r.tokens_out}")
        if r.chunks_retrieved is not None: extras.append(f"chunks={r.chunks_retrieved}")
        if r.answer_chars    is not None: extras.append(f"answer_chars={r.answer_chars}")
        suffix = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  ⏱  {r.label}: {r.elapsed_ms:.1f} ms{suffix}")

    def summary(self) -> None:
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        # Pipeline stages (non-QA)
        pipeline = [r for r in self.results if not r.label.startswith("QA:")]
        qa       = [r for r in self.results if r.label.startswith("QA:")]

        for r in pipeline:
            print(f"  {r.label:<30} {r.elapsed_ms:>8.1f} ms")

        if qa:
            times = [r.elapsed_ms for r in qa]
            print(f"\n  {'QA queries':<30} n={len(times)}")
            print(f"  {'  min':<30} {min(times):>8.1f} ms")
            print(f"  {'  max':<30} {max(times):>8.1f} ms")
            print(f"  {'  mean':<30} {statistics.mean(times):>8.1f} ms")
            if len(times) > 1:
                print(f"  {'  stdev':<30} {statistics.stdev(times):>8.1f} ms")
            total_in  = sum(r.tokens_in  for r in qa if r.tokens_in  is not None)
            total_out = sum(r.tokens_out for r in qa if r.tokens_out is not None)
            if total_in or total_out:
                print(f"  {'  total tokens in/out':<30} {total_in} / {total_out}")

        total = sum(r.elapsed_ms for r in self.results)
        print(f"\n  {'TOTAL':<30} {total:>8.1f} ms")
        print("=" * 60)


@contextmanager
def bench(session: BenchmarkSession, label: str, **kwargs):
    """Context manager that times a block and records the result."""
    t0 = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000
    session.record(BenchmarkResult(label=label, elapsed_ms=elapsed_ms, **kwargs))


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(pdf_path: str, session: BenchmarkSession):
    embeddings = OpenAIEmbeddings(
        check_embedding_ctx_length=False,
        model="text-embedding-qwen3-embedding-0.6b",
        api_key="not-needed",
        base_url="http://localhost:1234/v1",
    )

    llm = ChatOpenAI(
        model="local-model",
        api_key="not-needed",
        base_url="http://localhost:1234/v1",
    )

    with bench(session, "PDF load"):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

    print(f"  → {len(pages)} page(s) loaded")

    with bench(session, "Text splitting"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

    print(f"  → {len(chunks)} chunk(s) created")

    with bench(session, "Embedding + FAISS index"):
        vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain  = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain


def ask(qa_chain, question: str, session: BenchmarkSession) -> str:
    print(f"\n--- Question: {question} ---")

    t0 = time.perf_counter()
    response = qa_chain.invoke(question)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    answer = response.get("result", "No answer found.")

    # Token counts if the chain exposes them
    usage = response.get("usage_metadata") or {}
    tokens_in  = usage.get("input_tokens")
    tokens_out = usage.get("output_tokens")

    session.record(BenchmarkResult(
        label=f"QA: {question[:40]}{'…' if len(question) > 40 else ''}",
        elapsed_ms=elapsed_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        chunks_retrieved=3,       # fixed by search_kwargs k=3
        answer_chars=len(answer),
    ))

    print(f"Answer:\n{answer}")
    print("-" * 50)
    return answer


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Start\n")

    session  = BenchmarkSession()
    PDF_PATH = r"data\raw\cellar_439cd3a7-fd3c-4da7-8bf4-b0f60600c1d6.0004.02_DOC_1.pdf"

    qa_chain = build_pipeline(PDF_PATH, session)

    predefined_questions = [
        "Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. "
        "Als EU-burger wil ik graag weten op welke compensatie ik recht heb voor mijn economie klasse lucht?",
        "Mijn vlucht is geannuleerd, wat nu?",
        "Kan ik mijn stoelreservering upgraden naar business class?",
        "Ik heb recht op €400 compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?",
    ]

    for q in predefined_questions:
        ask(qa_chain, q, session)

    session.summary()


if __name__ == "__main__":
    main()