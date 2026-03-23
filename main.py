import re
import yaml
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

print("Start")

# Define your custom system prompt with reference instructions
SYSTEM_PROMPT = """Je bent een vriendelijke en behulpzame KLM klantenservice assistent die gespecialiseerd is in passagiersrechten en compensaties/terugbetalingen.

Gebruik de volgende context om de vraag van de klant te beantwoorden. Elke context sectie begint met een [Bron X: ...] label:

{context}

Belangrijke richtlijnen:
- Spreek de klant altijd aan met "u".
- Antwoord altijd in de taal van de vraag (Nederlands of Engels).
- VERPLICHT: Wees extreem specifiek over compensatiebedragen. Zoek in de context naar regels omtrent afstanden en uren om het exacte bedrag (bijv. €250, €400 of €600) te bepalen.
- Als een gebruiker specifieke steden noemt (zoals Istanbul en Amsterdam), schat dan zelf in onder welke afstandscategorie deze vlucht valt (bijv. meer dan 3500 km). Vraag de klant NIET om de afstand, maar bereken en noem direct het juiste bedrag.
- Als je het antwoord puur op basis van de context niet weet, zeg dat dan eerlijk.

KRITIEK BELANGRIJK - Bronvermelding regels:
1. Gebruik ALLEEN bronnummers die in de context staan (Bron 1, Bron 2, etc.).
2. Plaats [Bron X] inline DIRECT na het feit of de regel die je uit die bron haalt. Controleer extreem kritisch of het feit daadwerkelijk uit die specifieke bron komt voordat je de tag plaatst!
3. Voeg GEEN aparte "Bronnen:" sectie toe aan het einde; gebruik uitsluitend inline tags in de tekst.

Vraag: {question}

Antwoord:"""

embeddings = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    model="text-embedding-qwen3-embedding-0.6b",
    api_key="not-needed",
    base_url="http://localhost:1234/v1" 
)

llm = ChatOpenAI(
    model="local-model",
    api_key="not-needed",
    base_url="http://localhost:1234/v1"
)

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
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt template
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

# Keep this pure generation chain (no source post-processing yet)
generation_chain = prompt | llm | StrOutputParser()

def ask(question: str) -> str:
    docs = retriever.invoke(question)
    context = format_docs_with_sources(docs)
    raw_answer = generation_chain.invoke({"context": context, "question": question})
    return finalize_response_with_sources(raw_answer, docs)

def ask_and_print(question: str) -> None:
    print(f"\n--- Question: {question} ---")
    response = ask(question)
    print(f"Answer:\n{response}")
    print("-" * 50)

ask_and_print("Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. Als EU-burger wil ik graag weten op welke compensatie ik recht heb voor mijn economie klasse vlucht?")

ask_and_print("Mijn vlucht is geannuleerd, wat nu?")

ask_and_print("Kan ik mijn stoelreservering upgraden naar business class?")

ask_and_print("Ik heb recht op €400 compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?")

ask_and_print("Ik heb een vlucht van Brussel naar Amsterdam, maar mijn vlucht heeft 3 uur vertraging. Heb ik recht op compensatie?")

ask_and_print("Mijn vlucht heeft vertraging. Wat zijn mijn opties?")

ask_and_print("Vanaf welke vertragingstijd heb ik recht op compensatie?")

ask_and_print("Waar heb ik allemaal recht op als mijn vlucht vertraagd is?")

# ask_and_print(input("Ask something about your compensation: "))
