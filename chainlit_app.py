import os
import uuid
import re
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chainlit as cl
import redis

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and Redis URL from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize the ChatOpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initialize Redis client
redis_client = redis.Redis.from_url(REDIS_URL)


# Define a function to clean the text by removing extra spaces and non-ASCII characters
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


# Define a function to extract and clean English text from a PDF
def extract_and_clean_english_text(pdf_path, max_page=66):
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


# Initialize the embeddings
embd = OpenAIEmbeddings()

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)


# Define a function to split text into chunks
def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


# Define a function to get message history from Redis
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


### Contextualize question ###
# Define the system prompt for contextualizing a question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

### Answer question ###
# Define the system prompt for answering a question
system_prompt = (
    "You are an assistant for question-answering Tasks."
    "Use the following pieces of retrieved context to answer"
    "the question. If you don't know the answer, indicate that it is not mentioned in the context."
    "Provide a detailed answer if needed."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


@cl.on_chat_start
async def on_chat_start():
    # Prompt the user to upload a PDF file to begin
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    # Process the uploaded file
    file = files[0]

    msg = cl.Message(content=f"Processing {file.name}...", disable_feedback=True)
    await msg.send()

    with open(file.path, "rb") as f:
        pdf_path = file.path

    # Extract and clean English text from the PDF
    english_text = extract_and_clean_english_text(pdf_path)
    # Split the text into chunks
    text_chunks = split_text_into_chunks(english_text)

    # Create a vector store from the text chunks
    vectorstore = Chroma.from_texts(
        texts=text_chunks, collection_name="rag-chroma", embedding=embd
    )
    session_id = str(uuid.uuid4())

    # Set the retriever and session information in the user session
    retriever = vectorstore.as_retriever()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("pdf_path", pdf_path)
    cl.user_session.set("session_id", session_id)  # Store session ID

    msg.content = f"Processing {file.name} done. You can now ask questions!"
    await msg.update()

    # Send action buttons to create a new session and switch sessions
    existing_sessions = [
        session.decode() for session in redis_client.smembers("sessions")
    ]
    actions = [
        cl.Action(
            name="create_new_session",
            value="new_session",
            description="Create New Session",
        ),
        cl.Action(
            name="switch_session",
            value="switch_session",
            description="Switch Session",
        ),
    ]
    await cl.Message(
        content="You can create a new session or switch between existing sessions:",
        actions=actions,
    ).send()


@cl.action_callback("create_new_session")
async def on_create_new_session(action: cl.Action):
    # Generate a new session ID
    new_session_id = str(uuid.uuid4())
    cl.user_session.set(
        "session_id", new_session_id
    )  # Update session ID in the user session
    redis_client.sadd("sessions", new_session_id)
    print(f"New session ID created: {new_session_id}")

    return f"New session created with ID: {new_session_id}"


@cl.action_callback("switch_session")
async def on_switch_session(action: cl.Action):
    # List existing session IDs
    existing_sessions = [
        session.decode() for session in redis_client.smembers("sessions")
    ]

    # Prompt user to select a session
    if existing_sessions:
        actions = [
            cl.Action(
                name=session_id,
                value=session_id,
                description=f"Switch to session {session_id}",
            )
            for session_id in existing_sessions
        ]
        await cl.Message(
            content="Select a session to switch to:", actions=actions
        ).send()
    else:
        await cl.Message(content="No existing sessions found.").send()


@cl.action_callback("name")
async def on_select_session(action: cl.Action):
    # Switch to the selected session
    session_id = action.value
    cl.user_session.set("session_id", session_id)
    print(f"Switched to session ID: {session_id}")

    return f"Switched to session ID: {session_id}"


@cl.on_message
async def main(message: cl.Message):
    # Retrieve session ID from the user session
    session_id = cl.user_session.get("session_id")

    # Retrieve the retriever from the user session
    retriever = cl.user_session.get("retriever")

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create a conversational RAG chain with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Check if the question has been asked before
    question = message.content
    cached_response = redis_client.hget(f"session:{session_id}", question)

    if cached_response:
        # If cached response exists, return it
        await cl.Message(content=cached_response.decode()).send()
    else:
        # Set the session ID in the configuration
        config = {"configurable": {"session_id": session_id}}

        # Invoke the conversational RAG chain and get the result
        result = conversational_rag_chain.invoke({"input": question}, config=config)

        # Cache the result in Redis
        redis_client.hset(f"session:{session_id}", question, result["answer"])

        # Send the result back to the user
        await cl.Message(content=result["answer"]).send()


# Run the Chainlit app
if __name__ == "_main_":
    cl.run()
