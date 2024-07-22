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
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional
from langgraph.graph import END, StateGraph, START
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


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
            question: question
            generation: LLM generation
            documents: list of documents

    """

    question: str
    generation: str
    documents: List[str]


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


# Data model for hallucination grading
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Data model for answer grading
class GradeAnswer(BaseModel):
    """Binary score to assess if the answer addresses the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def grade_hallucinations(documents, generation):
    """
    Checks if the generation is grounded in the provided documents.

    Args:
        documents (str): The set of facts.
        generation (str): The LLM generated answer.

    Returns:
        str: 'yes' if the generation is grounded in the documents, otherwise 'no'.
        hallucination_grader: The initialized hallucination grader.
    """
    # Initialize hallucination grader
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Create hallucination grading prompt template
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    )

    # Combine prompt and grader
    hallucination_grader = hallucination_prompt | structured_llm_grader

    # Check if the generation is grounded in the provided documents
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    return grade, hallucination_grader


# function to grade answers
def grade_answer(question, generation):
    """
    Determines whether the generation addresses the user's question.

    Args:
        question (str): The user's question.
        generation (str): The LLM generated answer.

    Returns:
        str: 'yes' if the generation addresses the question, otherwise 'no'.
        answer_grader: The initialized answer grader.
    """
    # Initialize answer grader
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Create answer grading prompt template
    system = """You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "User question: \n\n {question} \n\n LLM generation: {generation}",
            ),
        ]
    )

    # Combine prompt and grader
    answer_grader = answer_prompt | structured_llm_grader

    # Check if the generation addresses the question
    score = answer_grader.invoke({"question": question, "generation": generation})
    grade = score.binary_score

    return grade, answer_grader


# Function to check if the generation is grounded in the document and answers the question
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers the question.

    Args:
        state (dict): The current graph state containing 'question', 'documents', and 'generation'.

    Returns:
        str: Decision for the next node to call ('useful', 'not useful', or 'not supported').
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check hallucination
    hallucination_grade, hallucination_grader = grade_hallucinations(
        documents, generation
    )

    if hallucination_grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        answer_grade, answer_grader = grade_answer(question, generation)

        if hallucination_grade == "yes":
            answer_grade, hallucination_grade = grade_answer(question, generation)
            if answer_grade == "yes":
                return "useful"
            else:
                return "not useful"

        else:
            return "not supported"


# re-write question if needed
def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with a re-phrased question.
    """
    question = state["question"]
    documents = state["documents"]

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create the prompt for question re-writing
    system = """You are a question re-writer that converts an input question to a better version that is optimized 
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    # Combine prompt and LLM
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # Re-write the question
    better_question = question_rewriter.invoke({"question": question})

    return {"documents": documents, "question": better_question}


# function to retrieve relevant docs from vectorstore
def retrieve(inputs: Dict[str, Any], vectorstore: Any) -> List[str]:
    """Retrieves relevant documents from the vectorstore."""
    question = inputs["question"]
    return vectorstore.similarity_search(question)


# function to generate answers
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


# Build a state
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
            question: question
            generation: LLM generation
            documents: list of documents

    """

    question: str
    generation: str
    documents: List[str]


# Create a workflow
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)


# Build edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {"not supported": "generate", "useful": END, "not useful": "transform_query"},
)

# Compile the workflow
app = workflow.compile()


from pprint import pprint

# Run
inputs = {
    "question": "What electrical supply and grounding requirements are necessary for the washing machine?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])
