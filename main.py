import re
import yaml
import json
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from typing import List, Union
from collections import deque

print("Start")

@dataclass
class QueryClassification:
    """Structured extraction of factual claims present in the user's query.
    
    Every field defaults to the 'absent' state so the rewriter knows
    exactly what was and was NOT stated by the user.
    """
    has_hours: bool = False
    hours_value: Optional[str] = None
    has_compensation_amount: bool = False
    compensation_value: Optional[str] = None
    asks_about_compensation_or_refund: bool = False
    has_origin: bool = False
    origin: Optional[str] = None
    has_destination: bool = False
    destination: Optional[str] = None

def merge_classification(
    base: QueryClassification, update: QueryClassification
) -> QueryClassification:
    """Merge new extraction into existing state.

    Only overwrites a field when the update extracted something (has_X=True).
    Allows corrections: a new non-null value for an existing field wins.
    Fields the update didn't find stay untouched from base.
    """
    return QueryClassification(
        has_hours=update.has_hours or base.has_hours,
        hours_value=update.hours_value if update.has_hours else base.hours_value,
        has_compensation_amount=update.has_compensation_amount or base.has_compensation_amount,
        compensation_value=update.compensation_value if update.has_compensation_amount else base.compensation_value,
        asks_about_compensation_or_refund=update.asks_about_compensation_or_refund or base.asks_about_compensation_or_refund,
        has_origin=update.has_origin or base.has_origin,
        origin=update.origin if update.has_origin else base.origin,
        has_destination=update.has_destination or base.has_destination,
        destination=update.destination if update.has_destination else base.destination,
    )

EXTRACTION_PROMPT = """You are a fact-extraction module. You receive a customer question about air travel / passenger rights.

Your ONLY job is to extract what the user LITERALLY stated. You must NEVER infer, assume, or add information that is not explicitly written.

## Current known state
The following JSON represents what has already been extracted from earlier messages in this conversation.
Your task is to return an UPDATED version of this JSON incorporating any NEW facts from the user's latest message.

Rules for updating:
- If a field already has a value and the user does NOT mention that topic, KEEP the existing value unchanged.
- If the user provides NEW information for a field, UPDATE that field.
- If the user CORRECTS a previously known value (e.g. "actually it's 4 hours"), UPDATE it to the new value.
- Only set a field to null / false if the user EXPLICITLY retracts it.

Current state:
{current_state}

## Extraction rules
Return a JSON object with exactly these keys:

{{
  "has_hours": true/false,
  "hours_value": "<digit(s) as string>" or null,
  "has_compensation_amount": true/false,
  "compensation_value": "<amount as string, e.g. '400 euro'>" or null,
  "asks_about_compensation_or_refund": true/false,
  "has_origin": true/false,
  "origin": "<city or country name>" or null,
  "has_destination": true/false,
  "destination": "<city or country name>" or null
}}

Rules:
1. has_hours is true ONLY if the user wrote a number (digit) followed by a time word (uur, uren, hours, hour, h). Written-out numbers like "drie uur" do NOT count — only digits like "3 uur" or "6 hours".
2. has_compensation_amount is true ONLY if the user mentions a specific monetary amount (e.g. "€400", "400 euro", "250 euros").
3. asks_about_compensation_or_refund is true if the user asks about compensation, vergoeding, compensatie, refund, terugbetaling, geld terug, or similar.
4. For origin/destination: determine which location is the departure and which is the arrival based on context words like "van", "vanuit", "from", "naar", "to", "richting". If the direction is ambiguous, set both has_origin and has_destination to false and leave values null.
5. Return ONLY the JSON object. No explanation, no markdown, no backticks.

User question: {question}"""

# Define your custom system prompt with reference instructions
SYSTEM_PROMPT = """Je bent een vriendelijke en behulpzame KLM klantenservice assistent die gespecialiseerd is in passagiersrechten en compensaties/terugbetalingen.

Gebruik de volgende context om de vraag van de klant te beantwoorden. Elke context sectie begint met een [Bron X: ...] label:

{context}

Belangrijke richtlijnen:
- Spreek de klant altijd aan met "u"
- Antwoord altijd in de taal van de vraag (Nederlands of Engels)
- Wees duidelijk en direct over compensatiebedragen en rechten
- Verwijs naar relevante EU verordeningen wanneer van toepassing
- Als je het antwoord niet weet op basis van de context, zeg dat eerlijk

KRITIEK BELANGRIJK - Bronvermelding regels:
1. Gebruik ALLEEN bronnummers die in de context staan (Bron 1, Bron 2, Bron 3, etc.)
2. Plaats [Bron X] inline direct na informatie uit die specifieke bron
3. Gebruik nooit een bronnummer dat niet in de context voorkomt
4. Voeg GEEN "Bronnen:" of "Sources:" sectie toe; alleen de antwoordtekst met inline [Bron X]

Conversatiegeschiedenis:
{history}

Vraag: {question}

Antwoord:"""

embeddings = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    model="text-embedding-qwen3-embedding-4b",
    api_key="not-needed",
    base_url="http://localhost:1234/v1" 
)

llm = ChatOpenAI(
    model="local-model",
    api_key="not-needed",
    base_url="http://localhost:1234/v1"
)

extraction_prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
extraction_chain = extraction_prompt | llm | StrOutputParser()

def load_markdown_with_metadata(file_path: Union[str, Path]) -> Document:
    """
    Load a markdown file with YAML frontmatter and extract metadata.
    
    Parses YAML frontmatter from the beginning of the file and extracts metadata.
    If no title is found in the frontmatter, uses the filename as the title.
    
    Parameters:
        file_path: Path to the markdown file, can be a string or Path object
    
    Returns:
        Document object with parsed content and metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metadata = {}
    body = content
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if frontmatter_match:
        try:
            metadata = yaml.safe_load(frontmatter_match.group(1)) or {}
        except yaml.YAMLError as e:
            print(f"Warning: could not parse frontmatter in {file_path}: {e}")
        body = content[frontmatter_match.end():]
    
    # Add the file name if no title exists
    if 'title' not in metadata:
        metadata['title'] = Path(file_path).stem

    return Document(page_content=body, metadata=metadata)

def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Format a list of documents with numbered source attribution labels.
    
    Creates formatted text blocks with source information extracted from document
    metadata. Supports URL sources, PDF page sources, or generic numbering.
    
    Parameters:
        docs: List of Document objects to format with source attribution
    
    Returns:
        Formatted string with documents separated by horizontal rules and
        prefixed with source labels in the format [Bron X: URL/info]
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        if 'url' in doc.metadata and doc.metadata['url']:
            source_info = f"[Bron {i}: {doc.metadata['url']}]"
        elif 'source' in doc.metadata:
            source_info = f"[Bron {i}: PDF pagina {doc.metadata.get('page', 'onbekend')}]"
        else:
            source_info = f"[Bron {i}]"
        
        formatted.append(f"{source_info}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)

def deduplicate_sources(response: str) -> str:
    """
    Post-process the response to remove duplicate sources in the Bronnen section.
    Keeps only the first occurrence of each unique URL and renumbers sources sequentially.
    Also updates inline citations to match the new numbering.
    """
    # Find the Bronnen section (supports both "Bronnen:" and "Sources:")
    bronnen_match = re.search(r'\n(Bronnen|Sources):\s*\n', response, re.IGNORECASE)
    
    if not bronnen_match:
        return response
    
    # Split response into main content and sources section
    split_pos = bronnen_match.start()
    main_content = response[:split_pos]
    sources_section = response[split_pos:]
    
    # Extract all source lines (numbered list items)
    source_pattern = r'^(\d+)\.\s+(.+)$'
    lines = sources_section.split('\n')
    
    seen_urls = {}  # Maps URL to its first occurrence number
    old_to_new_mapping = {}  # Maps old number to new number
    unique_sources = []
    header_line = None
    new_number = 1
    
    for line in lines:
        # Capture the header line (Bronnen: or Sources:)
        if re.match(r'^(Bronnen|Sources):\s*$', line, re.IGNORECASE):
            header_line = line
            continue
        
        # Check if it's a numbered source line
        match = re.match(source_pattern, line.strip())
        if match:
            old_number = int(match.group(1))
            url = match.group(2).strip()
            
            # Only add if we haven't seen this exact URL before
            if url not in seen_urls:
                seen_urls[url] = old_number
                old_to_new_mapping[old_number] = new_number
                unique_sources.append(f"{new_number}. {url}")
                new_number += 1
            else:
                # Map duplicate to the first occurrence's new number
                first_occurrence_old = seen_urls[url]
                old_to_new_mapping[old_number] = old_to_new_mapping[first_occurrence_old]
    
    # Update inline citations in main content
    updated_main_content = main_content
    for old_num, new_num in sorted(old_to_new_mapping.items(), reverse=True):
        # Use word boundaries to avoid replacing "1" in "10" or similar
        updated_main_content = re.sub(
            r'\[Bron\s+' + str(old_num) + r'\]',
            f'[Bron {new_num}]',
            updated_main_content,
            flags=re.IGNORECASE
        )
    
    # Reconstruct the response with deduplicated and renumbered sources
    if unique_sources and header_line:
        reconstructed_sources = f"\n\n{header_line}\n" + "\n".join(unique_sources)
        return updated_main_content + reconstructed_sources
    
    return response

def _collapse_duplicate_inline_citations(text: str) -> str:
    """
    Collapse repeated inline citations like:
    [Bron 1], [Bron 1] -> [Bron 1]
    [Bron 1] [Bron 1] -> [Bron 1]
    [Bron 1] en [Bron 1] -> [Bron 1]
    """
    pattern = re.compile(
        r"([Bron\s+\d+])(?:\s*,\s*|\s+en\s+|\s+)(\1)",
        flags=re.IGNORECASE
    )

    prev = None
    curr = text
    while curr != prev:
        prev = curr
        curr = pattern.sub(r"\1", curr)

    return curr

def _source_reference(doc: Document) -> str:
    """
    Build a stable human-readable source reference for one retrieved document.
    Priority: url > pdf source+page > title > fallback.
    """
    if doc.metadata.get("url"):
        return str(doc.metadata["url"])

    if doc.metadata.get("source"):
        page = doc.metadata.get("page")
        if page is not None:
            return f"{doc.metadata['source']} (pagina {page})"
        return str(doc.metadata["source"])

    if doc.metadata.get("title"):
        return str(doc.metadata["title"])

    return "Onbekende bron"


def _strip_model_sources_section(text: str) -> str:
    """
    Remove any model-generated Bronnen/Sources section.
    We generate sources ourselves to keep numbering consistent.
    """
    match = re.search(r"\n\s*(Bronnen|Sources)\s*:\s*\n", text, flags=re.IGNORECASE)
    if not match:
        return text.strip()
    return text[:match.start()].strip()


def _source_key_and_label(doc: Document) -> tuple:
    """
    Return a dedupe key + display label for a source.
    URL sources are deduped by normalized URL.
    Non-URL sources are deduped by their rendered reference.
    """
    url = doc.metadata.get("url")
    if url:
        url_text = str(url).strip()
        # Normalize for dedupe (case-insensitive, ignore trailing slash)
        url_key = url_text.rstrip("/").lower()
        return ("url", url_key), url_text

    label = _source_reference(doc).strip()
    return ("ref", label), label


def _extract_citation_numbers(text: str) -> List[int]:
    """
    Extract source numbers from citation blocks like:
    [Bron 1], [Bron 1, Bron 2, Bron 3], [Bron 1 en 2]
    """
    numbers: List[int] = []

    # Scan every [...] block and keep only ones that mention "Bron"
    for block in re.findall(r"\[([^\]]+)\]", text):
        if not re.search(r"\bbron\b", block, flags=re.IGNORECASE):
            continue
        for n in re.findall(r"\d+", block):
            numbers.append(int(n))

    return numbers


def _rewrite_citation_blocks(text: str, old_to_new: dict) -> str:
    """
    Rewrite citation blocks to normalized inline citations:
    [Bron 1, Bron 2] -> [Bron 1] [Bron 2]
    Invalid source ids are removed.
    """
    def repl(match: re.Match) -> str:
        block = match.group(1)

        if not re.search(r"\bbron\b", block, flags=re.IGNORECASE):
            return match.group(0)

        nums = [int(n) for n in re.findall(r"\d+", block)]
        mapped = []
        seen = set()

        for n in nums:
            if n in old_to_new:
                new_n = old_to_new[n]
                if new_n not in seen:
                    seen.add(new_n)
                    mapped.append(new_n)

        if not mapped:
            return ""

        return " ".join(f"[Bron {n}]" for n in mapped)

    return re.sub(r"\[([^\]]+)\]", repl, text)


def finalize_response_with_sources(answer: str, docs: List[Document]) -> str:
    """
    Make citations deterministic:
    - Remove model-written source list
    - Keep only valid source ids that exist in retrieved docs
    - Renumber in order of first appearance
    - Deduplicate by URL (fallback: reference label)
    - Always generate Bronnen when at least one valid citation exists
    """
    cleaned = _strip_model_sources_section(answer)

    found_numbers = _extract_citation_numbers(cleaned)

    valid_in_order = []
    seen_ids = set()
    for n in found_numbers:
        if 1 <= n <= len(docs) and n not in seen_ids:
            seen_ids.add(n)
            valid_in_order.append(n)

    old_to_new = {}
    source_key_to_new = {}
    deduped_labels = []

    for old_n in valid_in_order:
        key, label = _source_key_and_label(docs[old_n - 1])

        if key not in source_key_to_new:
            source_key_to_new[key] = len(deduped_labels) + 1
            deduped_labels.append(label)

        old_to_new[old_n] = source_key_to_new[key]

    normalized_body = _rewrite_citation_blocks(cleaned, old_to_new)
    normalized_body = _collapse_duplicate_inline_citations(normalized_body)
    normalized_body = re.sub(r"[ \t]{2,}", " ", normalized_body)
    normalized_body = re.sub(r"\s+([,.;:!?])", r"\1", normalized_body)
    normalized_body = normalized_body.strip()

    if not deduped_labels:
        return normalized_body

    source_lines = [f"{i}. {label}" for i, label in enumerate(deduped_labels, start=1)]
    return normalized_body + "\n\nBronnen:\n" + "\n".join(source_lines)

# Load documents
loader = PyPDFLoader("data\\raw\\cellar_439cd3a7-fd3c-4da7-8bf4-b0f60600c1d6.0004.02_DOC_1.pdf")
pages = loader.load()

md_docs = [load_markdown_with_metadata(p) for p in Path("data/processed").glob("*.md")]

all_docs = pages + md_docs

# Split and create vector store
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt template
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

# Keep this pure generation chain (no source post-processing yet)
generation_chain = prompt | llm | StrOutputParser()

REWRITE_PROMPT = """You are a query rewriter for a retrieval system about EU air passenger rights for the KLM airline.

Rewrite the user's question into a single precise retrieval query. You are NOT answering the question.

## Conversation history
{history}

## Pre-extracted facts from the user's query
The following facts were extracted from the user's original question. This is the GROUND TRUTH of what the user said. You MUST respect these exactly.

{extraction_block}

## Core rules
1. OUTPUT LANGUAGE: Write the rewritten query in the SAME language as the original question. Dutch in → Dutch out. English in → English out.
2. Preserve ALL factual details that appear in the extraction above: route, delay duration, airline name, flight class, dates, flight numbers, amounts, disruption type.
3. Add "EC 261/2004" only when the question is about delay, cancellation, denied boarding, or downgrade compensation.
4. Do NOT add facts, amounts, distances, or legal thresholds the user did not mention.
5. Do NOT add disruption types the user did not mention. If the user says "delay", do not add "cancellation" or "overbooking".
6. Remove greetings, emotional language, and filler that adds no factual content.
7. Return ONLY the rewritten query. No explanation, no preamble, no quotes.
8. ALWAYS ensure that the generated sentance makes sense and is grammatically correct, even if the user query was not.
9. Use conversation history to resolve references (e.g. "that flight", "same route", "what about cancellations?") but do NOT pull facts from history that the user did not repeat or reference in the current question.

## CRITICAL constraints from extraction
{constraint_block}

Original: {question}
Rewritten:"""

rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# --- Required fields that must be present before the RAG pipeline runs ---
REQUIRED_FIELDS = ["has_hours", "has_origin", "has_destination"]

# Prompt that asks the LLM to generate a natural follow-up question
# for the user, requesting the specific missing pieces of information.
FOLLOWUP_PROMPT = """You are a friendly KLM customer service assistant.
The customer asked a question but did not provide all necessary details.

Original question: {question}

The following information is still missing:
{missing_fields_description}

Write a short, polite follow-up message asking the customer to provide
the missing information. Write in the SAME language as the original question.
Do NOT answer the question yet. Do NOT guess or assume any values.
Return ONLY the follow-up message."""

followup_prompt = ChatPromptTemplate.from_template(FOLLOWUP_PROMPT)
followup_chain = followup_prompt | llm | StrOutputParser()


# Human-readable descriptions for each required field, in Dutch and English.
# Used to tell the LLM what to ask for.
FIELD_DESCRIPTIONS = {
    "has_hours": "the duration of the delay in hours (e.g. 3 hours, 6 hours)",
    "has_origin": "the departure city or airport",
    "has_destination": "the destination city or airport",
}

# Conversation history: stores up to 10 (question, answer) pairs, in-memory only
conversation_history: deque[dict] = deque(maxlen=10)


def format_history(history: deque[dict]) -> str:
    """Format history into a readable block for prompt injection."""
    if not history:
        return "Geen eerdere conversatie."
    lines = []
    for turn in history:
        lines.append(f"Klant: {turn['question']}")
        lines.append(f"Assistent: {turn['answer']}")
    return "\n".join(lines)

def get_missing_fields(clf: QueryClassification) -> list[str]:
    """Check which required fields are still absent in the classification.

    Returns a list of human-readable descriptions for every missing field.
    Empty list means the classification is complete.
    """
    missing = []
    for field_name in REQUIRED_FIELDS:
        if not getattr(clf, field_name):
            missing.append(FIELD_DESCRIPTIONS[field_name])
    return missing


def ask_followup(question: str, missing: list[str]) -> str:
    """Use the LLM to generate a natural follow-up question for the user.

    Takes the original question and a list of plain-language descriptions
    of what's missing, and returns a conversational prompt for the user.
    """
    missing_text = "\n".join(f"- {m}" for m in missing)
    return followup_chain.invoke({
        "question": question,
        "missing_fields_description": missing_text,
    })

def classify_query(
    question: str, current: Optional[QueryClassification] = None
) -> QueryClassification:
    """Extract factual claims from the user's query.

    When a current classification is provided, the LLM sees it as context
    and is instructed to preserve existing fields unless the user updates them.
    Falls back to an empty classification on parse failure.
    """
    # Serialize current state for the prompt (empty defaults if no prior state)
    if current is None:
        current = QueryClassification()

    current_state = json.dumps({
        "has_hours": current.has_hours,
        "hours_value": current.hours_value,
        "has_compensation_amount": current.has_compensation_amount,
        "compensation_value": current.compensation_value,
        "asks_about_compensation_or_refund": current.asks_about_compensation_or_refund,
        "has_origin": current.has_origin,
        "origin": current.origin,
        "has_destination": current.has_destination,
        "destination": current.destination,
    }, indent=2)

    raw = extraction_chain.invoke({
        "question": question,
        "current_state": current_state,
    })
    print(f"[classify_query] Raw extraction: {raw}")

    try:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[classify_query] JSON parse failed: {e} — returning current state unchanged")
        return current

    return QueryClassification(
        has_hours=bool(data.get("has_hours", False)),
        hours_value=data.get("hours_value"),
        has_compensation_amount=bool(data.get("has_compensation_amount", False)),
        compensation_value=data.get("compensation_value"),
        asks_about_compensation_or_refund=bool(data.get("asks_about_compensation_or_refund", False)),
        has_origin=bool(data.get("has_origin", False)),
        origin=data.get("origin"),
        has_destination=bool(data.get("has_destination", False)),
        destination=data.get("destination"),
    )

def build_extraction_block(clf: QueryClassification) -> str:
    """Build a human-readable summary of what the user DID state.
    
    This goes into the rewrite prompt so the rewriter sees the ground truth.
    """
    lines = []

    if clf.has_hours:
        lines.append(f"- Delay duration: {clf.hours_value} hours (USER STATED)")
    else:
        lines.append("- Delay duration: NOT mentioned by user")

    if clf.has_compensation_amount:
        lines.append(f"- Compensation amount: {clf.compensation_value} (USER STATED)")
    else:
        lines.append("- Compensation amount: NOT mentioned by user")

    if clf.asks_about_compensation_or_refund:
        lines.append("- Topic: User asks about compensation or refund (USER STATED)")
    else:
        lines.append("- Topic: User did NOT explicitly ask about compensation or refund")

    if clf.has_origin:
        lines.append(f"- Origin: {clf.origin} (USER STATED)")
    else:
        lines.append("- Origin: NOT mentioned by user")

    if clf.has_destination:
        lines.append(f"- Destination: {clf.destination} (USER STATED)")
    else:
        lines.append("- Destination: NOT mentioned by user")

    return "\n".join(lines)


def build_constraint_block(clf: QueryClassification) -> str:
    """Build explicit prohibitions for things the user did NOT say.
    
    These are injected as hard constraints into the rewrite prompt.
    """
    constraints = []

    if not clf.has_hours:
        constraints.append("- The user did NOT specify a number of hours. Do NOT add any hour amount to the rewritten query.")
    else:
        constraints.append(f"- The user specified {clf.hours_value} hours. Include exactly this value, do not change it.")

    if not clf.has_compensation_amount:
        constraints.append("- The user did NOT mention a compensation amount. Do NOT add any euro amount to the rewritten query.")
    else:
        constraints.append(f"- The user mentioned {clf.compensation_value}. Include exactly this value, do not change it.")

    if not clf.has_origin:
        constraints.append("- The user did NOT specify an origin/departure location. Do NOT add one.")
    else:
        constraints.append(f"- The user specified origin: {clf.origin}. Include it.")

    if not clf.has_destination:
        constraints.append("- The user did NOT specify a destination. Do NOT add one.")
    else:
        constraints.append(f"- The user specified destination: {clf.destination}. Include it.")

    return "\n".join(constraints)

def rewrite_query(
    question: str, clf: Optional[QueryClassification] = None
) -> str:
    """Rewrite the user question into a retrieval query.

    Accepts a pre-built classification to avoid redundant LLM calls.
    Falls back to classifying from scratch if none is provided.
    """
    if clf is None:
        clf = classify_query(question)
    print(f"[rewrite_query] Classification: {clf}")

    extraction_block = build_extraction_block(clf)
    constraint_block = build_constraint_block(clf)

    rewritten = rewrite_chain.invoke({
        "question": question,
        "extraction_block": extraction_block,
        "constraint_block": constraint_block,
        "history": format_history(conversation_history),
    })
    print(f"[rewrite_query] Rewritten question: {rewritten}")
    return rewritten


def ask(question: str) -> str:
    """Main entry point: classify, ask for missing info if needed, then answer.

    Maintains a running QueryClassification across follow-up rounds.
    The LLM receives the current state and returns a full updated JSON.
    merge_classification runs as a safety net in case the LLM drops a field.
    """
    accumulated_question = question

    # Initial extraction (no prior state)
    clf = classify_query(question)
    print(f"[ask] Initial classification: {clf}")

    while True:
        missing = get_missing_fields(clf)
        if not missing:
            break

        followup_message = ask_followup(accumulated_question, missing)
        print(f"\nAssistant: {followup_message}")

        user_reply = input("U: ")
        accumulated_question = f"{accumulated_question}\n{user_reply}"

        # LLM sees current state + new reply, returns full updated JSON
        llm_clf = classify_query(user_reply, current=clf)
        # Safety net: merge ensures the LLM didn't accidentally wipe fields
        clf = merge_classification(clf, llm_clf)
        print(f"[ask] Updated classification: {clf}")

    rewritten = rewrite_query(accumulated_question, clf=clf)
    docs = retriever.invoke(rewritten)
    context = format_docs_with_sources(docs)
    raw_answer = generation_chain.invoke({
        "context": context,
        "question": accumulated_question,
        "history": format_history(conversation_history),
    })
    final_answer = finalize_response_with_sources(raw_answer, docs)

    conversation_history.append({
        "question": question,
        "answer": final_answer,
    })

    return final_answer

def ask_and_print(question: str) -> None:
    print(f"\n--- Question: {question} ---")
    response = ask(question)
    print(f"Answer:\n{response}")
    print("-" * 50)

# --- Interactive terminal loop ---
# Runs until the user types "quit" or "exit".
# Each input starts a fresh conversation (the multi-turn follow-up
# loop for missing fields still happens inside ask()).
if __name__ == "__main__":
    print("\nKLM Passenger Rights Assistant")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("U: ").strip()

        # Allow the user to exit gracefully
        if user_input.lower() in ("quit", "exit", "q"):
            print("Tot ziens!")
            break

        # Skip empty input
        if not user_input:
            continue

        response = ask(user_input)
        print(f"\nAssistant: {response}\n")

# ask_and_print("Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. Als EU-burger wil ik graag weten op welke compensatie ik recht heb voor mijn economie klasse vlucht?")

# ask_and_print("Mijn vlucht is geannuleerd, wat nu?")

# ask_and_print("Kan ik mijn stoelreservering upgraden naar business class?")

# ask_and_print("Ik heb recht op €400 compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?")

# ask_and_print("Ik heb een vlucht van Brussel naar Amsterdam, maar mijn vlucht heeft 3 uur vertraging. Heb ik recht op compensatie?")

# ask_and_print("Mijn vlucht heeft vertraging. Wat zijn mijn opties?")

# ask_and_print("Vanaf welke vertragingstijd heb ik recht op compensatie?")

# ask_and_print("Waar heb ik allemaal recht op als mijn vlucht vertraagd is?")

# ask_and_print("Ik heb een vlucht van Brussel naar Amsterdam. Heb ik recht op compensatie?")

# ask_and_print(input("Ask something about your compensation: "))
