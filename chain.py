# chain.py
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from retrieval import get_retriever
from config import initialize_llm


# ── State ─────────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    answer: str
    grade: str
    retries: int



# ── Node Functions ────────────────────────────────────────
def rewrite_query(state: AgentState, llm) -> AgentState:
    """Rewrite the question for better retrieval."""
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Rewrite the following question 
        to be more specific and better suited for document retrieval.
        
        Original question: {question}
        
        Rewritten question:"""
    )
    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"question": state["question"]})
    return {**state, "rewritten_question": rewritten}


def retrieve(state: AgentState, retriever) -> AgentState:
    """Retrieve relevant documents."""
    docs = retriever.invoke(state["rewritten_question"])
    return {**state, "documents": docs}


def generate_answer(state: AgentState, llm) -> AgentState:
    """Generate an answer based on retrieved documents."""
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful university tutor for HSMW students.
        Answer the question based ONLY on the following lecture materials.
        If you cannot find the answer in the materials, say so clearly.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
    )
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["rewritten_question"],
    })
    return {**state, "answer": answer}


def grade_answer(state: AgentState, llm) -> AgentState:
    prompt = ChatPromptTemplate.from_template(
        """Is the following answer grounded in the provided context?
        Answer with only 'useful' or 'not useful'.

        Context:
        {context}

        Answer: {answer}"""
    )
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"context": context, "answer": state["answer"]}).strip().lower()

    if raw.startswith("useful"):
        grade = "useful"
    elif raw.startswith("not useful"):
        grade = "not useful"
    else:
        # fallback: be conservative
        grade = "not useful"

    return {**state, "grade": grade}



def should_retry(state: AgentState) -> str:
    """Decide whether to retry or finish."""
    if state["grade"] == "not useful":
        return "rewrite_query"
    return END


# ── Graph Builder ─────────────────────────────────────────
def build_graph(client):
    llm = initialize_llm(host=str(client._client.base_url), api_key=client._client.headers["Authorization"].replace("Bearer ", ""))
    retriever = get_retriever(client=client)

    # Bind dependencies into node functions
    def _rewrite(state): return rewrite_query(state, llm)
    def _retrieve(state): return retrieve(state, retriever)
    def _generate(state): return generate_answer(state, llm)
    def _grade(state): return grade_answer(state, llm)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("rewrite_query", _rewrite)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate_answer", _generate)
    graph.add_node("grade_answer", _grade)

    # Add edges
    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "grade_answer")
    graph.add_conditional_edges("grade_answer", should_retry)

    return graph.compile()
