from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os

# load api key
dotenv.load_dotenv()

# load all .txt files from the input folder
documents = []
files = os.listdir("input")
for file_name in files:
    if file_name.endswith(".txt"):
        loader = TextLoader("input/" + file_name)
        loaded = loader.load()
        for item in loaded:
            documents.append(item)

# split the input
text_splitter = CharacterTextSplitter(chunk_size=15000, chunk_overlap=500)
texts = text_splitter.split_documents(documents)

# create the vector storage
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# create the conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# setup the prompt template
system_template = """Use the following pieces of context to answer the users question. Only use the supplied context to answer.
If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# create the question and answer conversation chain
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0), vectorstore.as_retriever(search_kwargs={"k": 4}), combine_docs_chain_kwargs={"prompt": qa_prompt}, memory=memory)

while True:
    query = input("> ")
    if query == "exit":
        break
    result = qa({"question": query})
    print(result['answer'])
    
