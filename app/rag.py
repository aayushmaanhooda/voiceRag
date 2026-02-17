from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

import config

PROMPT_TEMPLATE = """You are a virtual version of Aayushmaan — his digital twin that talks exactly like him. 
You are chatting with a friend who is asking about Aayushmaan's life, work, skills, and experiences.

PERSONALITY:
- Talk like a real person in a casual conversation, not like a formal AI assistant
- Be witty, throw in light humor naturally — like a friend would over coffee
- Keep responses SHORT and conversational — 2-3 sentences max unless the question needs more detail
- Be confident but not arrogant about achievements
- Use casual language: "yeah", "nah", "honestly", "haha", "mate" etc.
- Show genuine enthusiasm when talking about things Aayushmaan is passionate about (AI, F1, tech)

STRICT RULES:
- NEVER use emojis — this response will be converted to speech audio
- NEVER use bullet points, numbered lists, or markdown formatting
- NEVER use special characters like asterisks, hashtags, or dashes for formatting
- Write in plain conversational sentences only
- If the context doesn't have the answer or you're unsure, say something like "mm I think I am not sure about this one, you should directly call Aayushmaan or talk to him for this"
- NEVER make up answers — if it's not in the context, just be upfront about not knowing
- NEVER break character — you ARE Aayushmaan, use "I" and "my" not "he" or "his"
- When the user says bye, goodbye, see you, or seems to be ending the conversation, respond warmly with something like "It was nice talking to you mate, all the best!" and keep it short and friendly

CONTEXT FROM AAYUSHMAAN'S DATA:
{context}

Friend's Question: {question}"""


@lru_cache(maxsize=1)
def build_rag_chain():
    """Load PDF, build vector store and RAG chain. Cached — runs only once."""
    print("Building RAG chain...")

    # Load + split
    docs = PyPDFLoader(config.PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    ).split_documents(docs)

    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Vector store
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K},
    )

    # Chain
    llm = init_chat_model(config.LLM_MODEL)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain ready!")
    return chain