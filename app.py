import os
import requests
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = "CourseAssistant/1.0"

# ── CONFIG ───────────────────────────────────────────────
PROGRAM_URLS = [
    "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",
    "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-admissions",      # ← fees & admissions
    "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-curriculum",      # ← full curriculum
    "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-career-prospects",# ← career outcomes
    "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-applynow",
]
DB_DIR = "./chroma_db"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="PGP AI Advisor | Masters' Union",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d0f18; }

/* Header */
.hero { padding: 2rem 0 1rem 0; }
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-sub { color: #6b7280; font-size: 1rem; margin-top: 6px; }

/* Metric cards */
.metrics-row { display: flex; gap: 16px; margin: 1.5rem 0; }
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #1a1d2e, #1e2235);
    border: 1px solid #2d3148;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 700; color: #f97316; }
.metric-lbl { font-size: 0.75rem; color: #6b7280; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }

/* Chat container */
.chat-wrap { max-width: 860px; margin: 0 auto; }

/* Messages */
[data-testid="stChatMessage"] {
    background: #13151f !important;
    border: 1px solid #1f2235 !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
    margin-bottom: 10px !important;
}

/* Input */
[data-testid="stChatInput"] textarea {
    background: #1a1d2e !important;
    border: 1.5px solid #f97316 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background: #0d0f18 !important; border-right: 1px solid #1f2235; }
.stButton > button {
    background: #13151f !important;
    color: #cbd5e1 !important;
    border: 1px solid #2d3148 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 10px 14px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #f97316 !important;
    color: #fff !important;
    border-color: #f97316 !important;
}

/* Info cards in sidebar */
.info-card {
    background: #13151f;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 12px;
}
.info-card-title { color: #f97316; font-weight: 600; font-size: 0.9rem; margin-bottom: 8px; }
.info-card-text { color: #94a3b8; font-size: 0.8rem; line-height: 1.6; }

/* Divider */
hr { border-color: #1f2235 !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── DATA INGESTION ───────────────────────────────────────
@st.cache_resource(show_spinner="🔍 Loading program data...")
def build_vectorstore():
    # Auto-download brochure if not present
    if not os.path.exists("brochure.pdf"):
        brochure_url = "https://files.mastersunion.link/MasterUnion/PGP_Applied_AI_and_Agentic_Systems_Brochure.pdf"
        resp = requests.get(brochure_url, timeout=60)
        with open("brochure.pdf", "wb") as f:
            f.write(resp.content)
    for url in PROGRAM_URLS:
        try:
            resp = requests.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "text/plain"},
                timeout=30
            )
            if resp.status_code == 200:
                # Strip nav boilerplate lines (pure markdown nav links)
                lines = resp.text.split('\n')
                filtered = [
                    line for line in lines
                    if not (line.strip().startswith('*   [') and line.strip().endswith(')'))
                    and not (line.strip().startswith('* [') and line.strip().endswith(')'))
                ]
                clean_text = '\n'.join(filtered)
                docs.append(Document(
                    page_content=clean_text,
                    metadata={"source": url}
                ))
        except Exception as e:
            st.warning(f"Could not scrape {url}: {e}")


    if os.path.exists("brochure.pdf"):
        docs += PyMuPDFLoader("brochure.pdf").load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

# ── RAG CHAIN ────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initializing AI...")
def build_chain(_vectordb):
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0)

    prompt = PromptTemplate.from_template("""
You are a helpful course advisor for the PGP in Applied AI and Agentic Systems by Masters' Union.
Answer ONLY using the context below. Be concise and student-friendly.
If asked about unrelated topics, politely redirect to the program.
If the answer is not in the context, say: "I don't have that information. Please contact ugadmissions@mastersunion.org or call +91 76691 86660."

Context:
{context}

Student Question: {question}

Answer:""")

    retriever = _vectordb.as_retriever(search_kwargs={"k": 8})
    def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, llm, retriever

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Course Advisor")
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("### 💡 Quick Questions")

    quick_qs = [
        "What is the fee structure?",
        "What are the eligibility criteria?",
        "What topics are in the curriculum?",
        "What are the career outcomes?",
        "How do I apply?",
        "What is the program duration?",
        "Who are the faculty members?",
        "What makes this program unique?",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">🤖 About This Assistant</div>
        <div class="info-card-text">Answers grounded in official Masters' Union brochure & website. No hallucinations.</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">📞 Contact Admissions</div>
        <div class="info-card-text">ugadmissions@mastersunion.org<br>+91 76691 86660</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">⚡ Powered By</div>
        <div class="info-card-text">LLaMA 3.3 70B · RAG Pipeline<br>ChromaDB · Sentence Transformers</div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 PGP Applied AI & Agentic Systems</div>
    <div class="hero-sub">Masters' Union · Intelligent Course Advisor · Powered by LLaMA 3.3 + RAG</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metrics-row">
    <div class="metric-card"><div class="metric-val">15 Months</div><div class="metric-lbl">Program Duration</div></div>
    <div class="metric-card"><div class="metric-val">Live AI</div><div class="metric-lbl">Powered Assistant</div></div>
    <div class="metric-card"><div class="metric-val">RAG</div><div class="metric-lbl">Grounded Answers</div></div>
    <div class="metric-card"><div class="metric-val">24 / 7</div><div class="metric-lbl">Always Available</div></div>
</div>
<hr>
""", unsafe_allow_html=True)

# ── LOAD RESOURCES ───────────────────────────────────────
vectordb = build_vectorstore()
chain, llm, retriever = build_chain(vectordb)

# ── CHAT ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! 👋 I'm your PGP Applied AI & Agentic Systems advisor. Ask me anything about curriculum, fees, admissions, or career outcomes!"
    }]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prefill = st.session_state.pop("prefill", None)

if query := (st.chat_input("Ask about fees, curriculum, admissions...", key="main_input") or prefill):
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        answer = chain.invoke(query)
        source_docs = retriever.invoke(query)
        sources = list(set(d.metadata.get("source", "Program Data") for d in source_docs))

    full_response = f"{answer}\n\n📚 *Source: {sources[0] if sources else 'Program Data'}*"
    st.chat_message("assistant").write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    with st.expander("💡 You might also want to ask..."):
        followups = llm.invoke(
            f"Based on this answer about a PGP AI program: '{answer[:300]}', "
            f"suggest exactly 3 short follow-up questions a student might ask. Numbered list only."
        ).content
        st.write(followups)
