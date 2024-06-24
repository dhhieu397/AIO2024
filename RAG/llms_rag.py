
# File pdf --> Load File --> Text Splitter --> Vectorization --> Vector Database

import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer , AutoModelForCausalLM
from langchain_huggingface . llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community . chat_message_histories import ChatMessageHistory
from langchain_community . document_loaders import PyPDFLoader, TextLoader
from langchain . chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core . runnables import RunnablePassthrough
from langchain_core . output_parsers import StrOutputParser
from langchain import hub

#Read file pdf:
loader = PyPDFLoader("YOLOv10_Tutorials.pdf")
documents = loader.load()
print(documents[0])

#Text Splitter
text_splitter = RecursiveCharacterTextSplitter( chunk_size =1000, chunk_overlap =100)
docs = text_splitter.split_documents(documents)
print("Number of sub - documents : ", len(docs))
print(docs[0])

from langchain.embeddings import HuggingFaceEmbeddings
#Instance vectorization
embedding = HuggingFaceEmbeddings()

#Instance Vector database
vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vector_db.as_retriever()
result = retriever.invoke('What is YOLO?')
print("Number of relevant documents : ", len(result))

nf4_config = BitsAndBytesConfig(bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

# Corrected model name, removing spaces and extra periods
MODEL_NAME = "lmsys/vicuna-7b-v1.5"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

from transformers import pipeline

# prompt: Integrate tokenizer and model into pipeline
model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    pad_token_id = tokenizer.eos_token_id,
    device_map="auto",

)
llm = HuggingFacePipeline(
  pipeline = model_pipeline)

prompt = hub.pull("rlm/rag-prompt")
def format_docs ( docs ) :
  return"\n\n".join(doc.page_content for doc in docs )

rag_chain = (
{"context": retriever | format_docs , "question": RunnablePassthrough() }  # Remove extra spaces around keys
| prompt
| llm
| StrOutputParser ()
)
USER_QUESTION = "YOLOv10 là gì?"
output = rag_chain.invoke(USER_QUESTION)
answer = output.split("Answer:")[1].strip()
print(answer)

#build chat interface
#chainlit

import chainlit as cl
import torch

from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

text_splitter = RecursiveCharacterTextSplitter ( chunk_size =1000 ,chunk_overlap =100)
embedding = HuggingFaceEmbeddings ()

def process_file(file: AskFileResponse):
  """
  Processes a file based on its type.

  Args:
    file: An AskFileResponse object containing the file information.

  Returns:
    A list of documents extracted from the file.
  """

  if file.type == "text/plain":
    Loader = TextLoader
  elif file.type == "application/pdf":
    Loader = PyPDFLoader
  else:
    raise ValueError("Unsupported file type.")

  loader = Loader(file.path)
  return loader.load()


#instance chroma db
def get_vector_db(file: AskFileResponse):
  """
  Creates a vector database from a file.

  Args:
    file: An AskFileResponse object containing the file information.

  Returns:
    A Chroma vector database.
  """

  docs = process_file(file)
  cl.user_session.set("docs", docs)
  vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
  return vector_db

# prompt: format this code: 'def get_huggingface_llm ( model_name : str = "lmsys / vicuna -7b-v1 .5",
# 2 max_new_token : int = 512) :
# 3 nf4_config = BitsAndBytesConfig (
# 4 load_in_4bit =True ,
# 5 bnb_4bit_quant_type ="nf4",
# 9
# AI VIETNAM (AIO2024) aivietnam.edu.vn
# 6 bnb_4bit_use_double_quant =True ,
# 7 bnb_4bit_compute_dtype = torch . bfloat16
# 8 )
# 9 model = AutoModelForCausalLM . from_pretrained (
# 10 model_n

def get_huggingface_llm(
    model_name: str = "lmsys/vicuna-7b-v1.5",
    max_new_token: int = 512,
) -> pipeline:
    """
    Loads a HuggingFace LLM with quantization and returns a pipeline.

    Args:
        model_name: The name of the HuggingFace model to load.
        max_new_token: The maximum number of new tokens to generate.

    Returns:
        A pipeline for the loaded LLM.
    """

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
)
    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
)

    return llm
LLM = get_huggingface_llm()

welcome_message = """Welcome to the PDF QA! To get started :
1. Upload a PDF or text file
2. Ask a question about the file
"""

@cl.on_chat_start
async def on_chat_start():
  files = None
  while files is None:
    files = await cl.AskFileMessage(
        content=welcome_message,
        accept=["text/plain", "application/pdf"],
        max_size_mb=20,
        timeout=180).send()
  file = files[0]
  # Define the msg variable here
  msg = cl.Message(content=f'Processing file {file.name}...', disable_feedback=True)
  await msg.send()
  vector_db = await cl.make_async(get_vector_db)(file)

  message_history = ChatMessageHistory()
  memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', chat_memory=message_history, return_messages=True)
  retirever = vector_db.as_retriever(search_type='mmr', search_kwargs={'k': 3})

  chain = ConversationalRetrievalChain.from_llm(llm =LLM, chain_type ="stuff", retriever = retriever, memory =memory, return_source_documents = True)

  msg.content = f" ‘{ file . name } ‘ processed . You can now ask questions !" # Update the content of the msg variable
  await msg.update () # Now this line should work
  cl.user_session.set("chain", chain )

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    print(source_documents)
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
          print(source_doc)
          source_name = f"source_{source_idx}"
          text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
          answer += f"\nSources: {', '.join(source_names)}"
        else:
          answer += "\nNosource found"
    await cl.Message(content=answer, elements=text_elements).send()
