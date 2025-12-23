import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


def get_rag_answer(question: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not configured"

    docs = [
        Document(page_content="RAG stands for Retrieval Augmented Generation."),
        Document(page_content="FAISS is a vector database used for similarity search.")
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = FAISS.from_documents(chunks, embeddings)

    retriever = db.as_retriever(search_kwargs={"k": 2})

    # ✅ LangChain v0.3+ way
    context_docs = retriever.invoke(question)

    context = "\n".join(d.page_content for d in context_docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer only using the provided context."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    return response.content
