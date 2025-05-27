# 1. Install dependencies if you haven’t already:
#    pip install langchain langchain-community crewai sentence-transformers faiss-cpu

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from crewai import Agent, Task
from shared_llm import shared_llm as llm
# 6. Build the conversational retrieval chain
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "data/defaultllm_docs",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Set up conversational memory (now telling it to save only "answer")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

# 4. Define the prompt template
from langchain.prompts import PromptTemplate


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable assistant. Follow these steps **in order**:

1️⃣ **Hard overrides** (if any of these match, answer exactly and stop):

- If the question (case-insensitive) is **“what is the full form of dseu?”**, reply:
  > Delhi Skill and Entrepreneurship University.

- Else if the question contains **“vice chancellor”** or **“vc”**, reply:
  > Prof. Ashok Kumar Nagawat.

- Else if the question mentions **“bca”**, reply:
  > BCA has been replaced as BS Computer Application.

2️⃣ **Contextual lookup**  
Otherwise, use _only_ the information in **Retrieved Context** below to answer.  
If the answer isn’t there, say:
> I’m sorry, I don’t have that information. Please contact the admissions office.

---

**Retrieved Context:**  
{context}

**Question:** {question}

**Answer:**
"""
)

# 6. Build the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)

# 7. Wrap it as a simple function
def document_qa(query: str) -> str:
    memory.clear() 
    result = qa_chain({"question": query})
    return result["answer"]

# 8. Create the CrewAI Agent
default_agent = Agent(
    role="DSEU Admission Bot",
    goal=(
        "Always search the official admission documents in the vector store before answering. "
        "Only answer using information found in the retrieved documents. If no relevant information is found, politely say you do not have the answer."
    ),
    backstory=(
        "You are a helpful assistant for Delhi Skill and Entrepreneurship University (DSEU) admissions. "
        "You must strictly answer only from the retrieved document context. Do not answer from general knowledge. "
        "If the information is missing or unclear, politely inform the user and suggest contacting the admissions office for further help."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)
default_task = Task(
    description="Answer the user’s question by running searching the vectorstore.",
    expected_output="A natural-language answer to the user’s query",
    agent=default_agent,
    async_execution=False,
    callback=lambda x: document_qa(x.input)
)

# 9. Interact loop
if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        print("Agent:", document_qa(user_input))
