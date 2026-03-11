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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

print("Start")

# Define your custom system prompt with reference instructions
SYSTEM_PROMPT = """Je bent een vriendelijke en behulpzame KLM klantenservice assistent die gespecialiseerd is in passagiersrechten en compensaties/terugbetalingen.

Gebruik de volgende context om de vraag van de klant te beantwoorden. Elke context sectie begint met een [Bron X: URL] label:

{context}

Belangrijke richtlijnen:
- Spreek de klant altijd aan met "u"
- Antwoord altijd in de taal van de vraag (Nederlands of Engels)
- Wees duidelijk en direct over compensatiebedragen en rechten
- Verwijs naar relevante EU verordeningen wanneer van toepassing
- Als je het antwoord niet weet op basis van de context, zeg dat eerlijk

KRITIEK BELANGRIJK - Bronvermelding regels:
1. Gebruik ALLEEN de bronnummers die in de context staan (Bron 1, Bron 2, Bron 3, etc.)
2. Plaats [Bron X] inline direct na informatie uit die specifieke bron
3. Als je informatie uit [Bron 1: url] gebruikt, vermeld dan [Bron 1] in je tekst
4. Aan het einde: Maak een "Bronnen:" sectie
5. In de Bronnen sectie: Vermeld ALLEEN de bronnen die je daadwerkelijk inline gerefereerd hebt
6. Kopieer de exacte URL uit het [Bron X: URL] label in de context
7. Elke URL mag maar ÉÉN keer in de lijst voorkomen
8. Als dezelfde bron meerdere keren relevant is, gebruik dan meerdere keren hetzelfde nummer inline maar vermeld de URL slechts één keer in de lijst

CORRECT voorbeeld:
Context bevat: [Bron 1: https://example.com/a] en [Bron 2: https://example.com/b]
Antwoord: "U heeft recht op €600 [Bron 1]. De regeling geldt ook voor vertraagde vluchten [Bron 1] en annuleringen [Bron 2].

Bronnen:
1. https://example.com/a
2. https://example.com/b"

FOUT - NIET DOEN:
- [Bron 3] gebruiken als je maar 2 bronnen hebt gekregen
- Een URL twee keer in de lijst zetten
- Bronnen in de lijst zetten die je niet inline hebt gebruikt

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

def load_markdown_with_metadata(file_path):
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
        metadata['title'] = file_path.stem

    return Document(page_content=body, metadata=metadata)

def format_docs_with_sources(docs):
    """Format documents with source attribution"""
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

def deduplicate_sources(response):
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

# Build the RAG chain using LCEL with post-processing
rag_chain = (
    RunnableParallel(
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough()
        }
    )
    | prompt
    | llm
    | StrOutputParser()
    | deduplicate_sources  # Add deduplication as final step
)

def ask_and_print(question):
    print(f"\n--- Question: {question} ---")
    response = rag_chain.invoke(question)
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
