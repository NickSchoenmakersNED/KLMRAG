from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

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

loader = PyPDFLoader("data\\raw\cellar_439cd3a7-fd3c-4da7-8bf4-b0f60600c1d6.0004.02_DOC_1.pdf")
pages = loader.load() 

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

ask_and_print("Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. Als EU-burger wil ik graag weten op welke compensatie ik recht heb voor mijn economie klasse lucht?")

ask_and_print("Mijn vlucht is geannuleerd, wat nu?")

ask_and_print("Kan ik mijn stoelreservering upgraden naar business class?")

ask_and_print("Ik heb recht op €400 compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?")

ask_and_print(input("Ask something about your compensation: "))
