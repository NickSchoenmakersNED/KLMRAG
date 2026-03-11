import re
import yaml
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

print("Start")

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

def ask_and_print(question):
    print(f"\n--- Question: {question} ---")
    response = qa_chain.invoke(question)
    
    text_result = response.get('result', 'No answer found.')
    
    print(f"Answer:\n{text_result}")
    print("-" * 50)

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

    return Document(page_content=body, metadata=metadata)

loader = PyPDFLoader("data\\raw\cellar_439cd3a7-fd3c-4da7-8bf4-b0f60600c1d6.0004.02_DOC_1.pdf")
pages = loader.load()

md_docs = [load_markdown_with_metadata(p) for p in Path("data/processed").glob("*.md")]

all_docs = pages + md_docs

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

ask_and_print("Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. Als EU-burger wil ik graag weten op welke compensatie ik recht heb voor mijn economie klasse vlucht?")

ask_and_print("Mijn vlucht is geannuleerd, wat nu?")

ask_and_print("Kan ik mijn stoelreservering upgraden naar business class?")

ask_and_print("Ik heb recht op €400 compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?")

ask_and_print("Ik heb een vlucht van Brussel naar Amsterdam, maar mijn vlucht heeft 3 uur vertraging. Heb ik recht op compensatie?")

ask_and_print("Mijn vlucht heeft vertraging. Wat zijn mijn opties?")

ask_and_print("Vanaf welke vertragingstijd heb ik recht op compensatie?")

ask_and_print("Waar heb ik allemaal recht op als mijn vlucht vertraagd is?")

ask_and_print(input("Ask something about your compensation: "))

## ask_and_print(input("Ask something about your compensation: "))
