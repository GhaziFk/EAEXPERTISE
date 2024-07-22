import fitz  # PyMuPDF
from langdetect import detect
import re
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional
import jwt
import chainlit as cl

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "EAEXPERTISE"
os.environ["LANGCHAIN_API_KEY"] = OPENAI_API_KEY


# Utility functions
def clean_text(text: str) -> str:
    """Cleans the extracted text."""
    text = re.sub(r"\s+", " ", text)  # Replace multiple whitespace
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
    return text.strip()


def extract_and_clean_english_text(pdf_path: str, max_page: int = 66) -> str:
    """Extracts and cleans English text from a PDF."""
    pdf_document = fitz.open(pdf_path)
    english_text = []

    for page_num in range(min(len(pdf_document), max_page)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        lines = re.split(r"\n", text)

        for line in lines:
            try:
                if detect(line) == "en":
                    cleaned_line = clean_text(line)
                    if cleaned_line:
                        english_text.append(cleaned_line)
            except:
                continue

    return " ".join(english_text)


def split_text_into_chunks(
    text: str, chunk_size: int = 2000, chunk_overlap: int = 350
) -> List[str]:
    """Splits text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


def setup_vectorstore(text_chunks: List[str]) -> Any:
    """Sets up the vectorstore with the given text chunks."""
    embd = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=text_chunks, collection_name="rag-chroma", embedding=embd
    )
    return vectorstore


def create_llm_chain() -> Any:
    """Creates an LLM generation chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    template = """
    You are a helpful assistant. Answer the question based only on the following context:
    {context}
    Answer the question based on the above question: {question}
    Provide a detailed answer.
    Don't justify your answers.
    Don't give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    If you don't know the answer, don't hallucinate. Just say: 'I can't answer this question since it is not mentioned in the context.'
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


def retrieve(inputs: Dict[str, Any], vectorstore: Any) -> List[str]:
    """Retrieves relevant documents from the vectorstore."""
    question = inputs["question"]
    return vectorstore.similarity_search(question)


def generate(state: dict) -> dict:
    """
    Generate an answer based on the provided context and question.

    Args:
        state (dict): The current graph state containing 'question' and 'documents'.

    Returns:
        state (dict): Updated state with a new key 'generation' containing the LLM's answer.
    """

    question = state["question"]
    documents = state["documents"]

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create a custom prompt template
    template = """ 
    You are a helpful assistant, Answer the question based only on the following context :
    {context}
    Answer the question based on the above question: {question}
    Provide a detailed answer.
    Don't justify your answers.
    Don't give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    if you don't know the answer, don't hallucinate. Just say: 'I can't answer this question since it is not mentioned in the context.'
    """

    # Initialize the ChatPromptTemplate with the template
    prompt = ChatPromptTemplate.from_template(template)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # RAG generation
    generation = rag_chain.invoke(
        {"context": format_docs(documents), "question": question}
    )

    # Update the state with the generated answer
    return {"documents": documents, "question": question, "generation": generation}


@cl.on_chat_start
async def on_chat_start():
    pdf_path = "washer.pdf"
    cleaned_text = extract_and_clean_english_text(pdf_path)
    text_chunks = split_text_into_chunks(cleaned_text)
    vectorstore = setup_vectorstore(text_chunks)
    cl.user_session.set("vectorstore", vectorstore)
    await cl.Message(content="Welcome! Ask me anything about the washer manual.").send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content
    vectorstore = cl.user_session.get("vectorstore")
    inputs = {"question": question}
    documents = retrieve(inputs, vectorstore)
    state = {"documents": documents, "question": question}
    output = generate(state)
    await cl.Message(content=output["generation"]).send()


# Run the Chainlit app
if __name__ == "__main__":
    cl.run()
