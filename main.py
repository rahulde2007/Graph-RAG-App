import streamlit as st
from dotenv import load_dotenv
import os, tempfile, json
from uuid import uuid4

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_PASS   = os.getenv("NEO4J_PASSWORD")
URI          = "neo4j+ssc://84635adc.databases.neo4j.io"
USERNAME     = "84635adc"
DB_NAME      = "84635adc"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from groq import Groq as GroqDirect
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Graph RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:           #0B0912;
    --bg-sidebar:   #0F0C18;
    --bg-card:      #161126;
    --bg-hover:     #1E1833;
    --bg-input:     rgba(255,255,255,0.04);
    --accent:       #8B5CF6;
    --accent-2:     #A78BFA;
    --accent-glow:  rgba(139,92,246,0.2);
    --accent-soft:  rgba(139,92,246,0.06);
    --border:       rgba(255,255,255,0.06);
    --border-focus: rgba(139,92,246,0.5);
    --text-1:       #F4F0FF;
    --text-2:       #9B93B0;
    --text-3:       #5C5573;
    --success:      #34D399;
    --danger:       #F87171;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background: var(--bg) !important; }
.stApp {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(139,92,246,0.12) 0%, var(--bg) 60%) !important;
    color: var(--text-1);
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }

[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 20px !important; }
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-3) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] label { color: var(--text-2) !important; font-size: 13px !important; }
[data-testid="stSidebar"] h2 {
    color: var(--text-1) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 1px dashed rgba(139,92,246,0.25) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,92,246,0.5) !important;
    background: var(--accent-soft) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 24px rgba(139,92,246,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #7C3AED) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-2) !important;
    box-shadow: none !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--bg-hover) !important;
    color: var(--text-1) !important;
    transform: none !important;
    box-shadow: none !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-3) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-2) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 28px 0 !important; }

[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 6px 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(139,92,246,0.06) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    border: 1px solid rgba(139,92,246,0.15) !important;
}
[data-testid="chatAvatarIcon-user"] { background: rgba(255,255,255,0.06) !important; border-radius: 8px !important; }
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #7C3AED, #A78BFA) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 16px rgba(139,92,246,0.3) !important;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    color: var(--text-1) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-1) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-3) !important; }
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    border-radius: 10px !important;
}

[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stAlert { border-radius: 10px !important; }
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* Checkboxes */
[data-testid="stCheckbox"] label { color: var(--text-2) !important; font-size: 13px !important; }
[data-testid="stCheckbox"] input:checked + div { background: var(--accent) !important; border-color: var(--accent) !important; }

/* Document card */
.doc-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.15s;
}
.doc-card:hover { background: rgba(255,255,255,0.05); border-color: rgba(139,92,246,0.2); }
.doc-card.active { border-color: rgba(139,92,246,0.4); background: rgba(139,92,246,0.06); }
.doc-name { font-size: 12px; color: var(--text-1); font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 130px; }
.doc-meta { font-size: 10px; color: var(--text-3); font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.doc-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 6px var(--accent); flex-shrink: 0; }

.stat-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 11px; color: var(--text-3);
    margin: 3px 3px 0 0;
    font-family: 'JetBrains Mono', monospace;
}
.stat-pill .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 6px var(--accent); }

/* Source badge */
.source-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    color: var(--accent-2);
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 3px 0 0;
}

/* Cross-doc banner */
.cross-doc-banner {
    background: rgba(139,92,246,0.06);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 12px;
    color: var(--accent-2);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.orb-container { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 70px 40px 80px; text-align: center; }
.orb-wrap { position: relative; width: 96px; height: 96px; margin-bottom: 28px; }
.orb { width: 96px; height: 96px; border-radius: 50%;
    background: conic-gradient(from 180deg at 50% 50%, #7C3AED 0deg, #A78BFA 90deg, #C4B5FD 150deg, #7C3AED 210deg, #5B21B6 270deg, #7C3AED 360deg);
    box-shadow: 0 0 40px rgba(139,92,246,0.4), 0 0 80px rgba(139,92,246,0.15), inset 0 0 30px rgba(255,255,255,0.1);
    animation: orb-rotate 6s linear infinite;
}
.orb::after { content: ''; position: absolute; inset: 0; border-radius: 50%; background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.25) 0%, transparent 60%); }
.orb-glow { position: absolute; inset: -16px; border-radius: 50%; background: radial-gradient(circle, rgba(139,92,246,0.2) 0%, transparent 70%); animation: orb-pulse 3s ease-in-out infinite; }
@keyframes orb-rotate { from { filter: hue-rotate(0deg); } to { filter: hue-rotate(360deg); } }
@keyframes orb-pulse { 0%, 100% { transform: scale(1); opacity: 0.6; } 50% { transform: scale(1.15); opacity: 1; } }
.orb-title { font-size: 22px; font-weight: 500; color: var(--text-1); letter-spacing: -0.02em; margin-bottom: 10px; }
.orb-sub { font-size: 14px; color: var(--text-3); max-width: 300px; line-height: 1.6; }

.graph-label { font-size: 11px; color: var(--text-3); font-family: 'JetBrains Mono', monospace; padding: 10px 14px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; text-transform: uppercase; letter-spacing: 0.04em; }

.step-item { display: flex; align-items: center; gap: 10px; padding: 7px 0; color: var(--text-3); font-size: 13px; }
.step-num { width: 18px; height: 18px; border-radius: 50%; border: 1px solid rgba(139,92,246,0.3); display: flex; align-items: center; justify-content: center; font-size: 10px; color: var(--accent-2); flex-shrink: 0; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────
if "documents"    not in st.session_state: st.session_state.documents    = {}  # id → doc data
if "active_docs"  not in st.session_state: st.session_state.active_docs  = []  # selected doc ids
if "messages"     not in st.session_state: st.session_state.messages     = []
if "embeddings"   not in st.session_state: st.session_state.embeddings   = None
if "llm"          not in st.session_state: st.session_state.llm          = None

# ── Helper: Init shared resources ─────────
def get_llm():
    if not st.session_state.llm:
        st.session_state.llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b")
    return st.session_state.llm

def get_embeddings():
    if not st.session_state.embeddings:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return st.session_state.embeddings

# ── Sidebar ───────────────────────────────
with st.sidebar:
    st.markdown("## Graph RAG")
    st.markdown("&nbsp;")

    # ── Upload new PDF ──
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        # check if already uploaded
        existing_names = [d["name"] for d in st.session_state.documents.values()]
        if uploaded_file.name in existing_names:
            st.warning("Already uploaded!")
        else:
            if st.button("Add Document", use_container_width=True):
                doc_id = str(uuid4())[:8]  # short unique id
                with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:

                    st.write("Loading PDF...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    st.write(f"✓ {len(documents)} pages")

                    st.write("Splitting chunks...")
                    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
                    chunks = splitter.split_documents(documents)
                    # tag each chunk with doc_id
                    for chunk in chunks:
                        chunk.metadata["doc_id"] = doc_id
                        chunk.metadata["doc_name"] = uploaded_file.name
                    st.write(f"✓ {len(chunks)} chunks")

                    st.write("Building Knowledge Graph...")
                    llm = get_llm()
                    graph_db = Neo4jGraph(url=URI, username=USERNAME, password=NEO4J_PASS, database=DB_NAME)
                    transformer = LLMGraphTransformer(llm=llm)
                    graph_docs = transformer.convert_to_graph_documents(chunks)
                    graph_db.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

                    nodes, edges = [], []
                    node_ids = set()
                    for doc in graph_docs:
                        for n in doc.nodes:
                            if n.id not in node_ids:
                                nodes.append({"id": n.id, "label": n.type or "Entity"})
                                node_ids.add(n.id)
                        for r in doc.relationships:
                            edges.append({"from": r.source.id, "to": r.target.id, "label": r.type})
                    st.write(f"✓ {len(nodes)} nodes, {len(edges)} relations")

                    st.write("Creating embeddings...")
                    embeddings = get_embeddings()
                    # each document gets its own vector index
                    vector_store = Neo4jVector.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        url=URI,
                        username=USERNAME,
                        password=NEO4J_PASS,
                        database=DB_NAME,
                        index_name=f"vector_{doc_id}",
                        node_label=f"Chunk_{doc_id}",
                    )
                    st.write("✓ Vectors stored")

                    os.unlink(tmp_path)

                    # Save to session state
                    st.session_state.documents[doc_id] = {
                        "name": uploaded_file.name,
                        "id": doc_id,
                        "vector_store": vector_store,
                        "graph_db": graph_db,
                        "nodes": nodes,
                        "edges": edges,
                        "stats": {"nodes": len(nodes), "rels": len(edges), "chunks": len(chunks), "pages": len(documents)}
                    }

                    # Auto-select new document
                    if doc_id not in st.session_state.active_docs:
                        st.session_state.active_docs.append(doc_id)

                    status.update(label="Done!", state="complete")
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Document Library ──
    if st.session_state.documents:
        st.markdown("**Documents**")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        for doc_id, doc in st.session_state.documents.items():
            is_active = doc_id in st.session_state.active_docs
            col1, col2 = st.columns([5, 1])
            with col1:
                checked = st.checkbox(
                    doc["name"][:22] + ("…" if len(doc["name"]) > 22 else ""),
                    value=is_active,
                    key=f"check_{doc_id}"
                )
                if checked and doc_id not in st.session_state.active_docs:
                    st.session_state.active_docs.append(doc_id)
                    st.rerun()
                elif not checked and doc_id in st.session_state.active_docs:
                    st.session_state.active_docs.remove(doc_id)
                    st.rerun()

                stats = doc["stats"]
                st.markdown(f"""
                <div style="font-size:10px;color:#5C5573;font-family:'JetBrains Mono',monospace;
                            margin-top:-8px;padding-bottom:4px">
                    {stats['pages']}p · {stats['nodes']}n · {stats['rels']}r
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if st.button("✕", key=f"del_{doc_id}", help="Remove document"):
                    del st.session_state.documents[doc_id]
                    if doc_id in st.session_state.active_docs:
                        st.session_state.active_docs.remove(doc_id)
                    st.session_state.messages = []
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Select all / None
        col1, col2 = st.columns(2)
        with col1:
            if st.button("All", use_container_width=True):
                st.session_state.active_docs = list(st.session_state.documents.keys())
                st.rerun()
        with col2:
            if st.button("None", use_container_width=True):
                st.session_state.active_docs = []
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-item"><span class="step-num">1</span>Upload PDFs</div>
    <div class="step-item"><span class="step-num">2</span>Select documents to query</div>
    <div class="step-item"><span class="step-num">3</span>Ask anything</div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────
doc_count = len(st.session_state.documents)
active_count = len(st.session_state.active_docs)

st.markdown(f"""
<div style="padding:32px 0 18px; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:20px">
    <div style="font-size:24px;font-weight:600;color:#F4F0FF;letter-spacing:-0.03em;
                display:flex;align-items:center;gap:10px">
        <span style="width:9px;height:9px;border-radius:50%;
                     background:linear-gradient(135deg,#7C3AED,#A78BFA);
                     box-shadow:0 0 12px rgba(139,92,246,0.6);
                     display:inline-block;flex-shrink:0"></span>
        Graph RAG
    </div>
    <div style="font-size:13px;color:#5C5573;margin-top:5px">
        {doc_count} document{"s" if doc_count != 1 else ""} · {active_count} selected
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────
tab_chat, tab_graph = st.tabs(["💬  Chat", "🕸️  Knowledge Graph"])

# ── TAB 1: CHAT ───────────────────────────
with tab_chat:

    def get_graph_context(question, doc_ids):
        if not doc_ids:
            return ""
        # Use first active doc's graph_db
        graph_db = st.session_state.documents[doc_ids[0]]["graph_db"]
        entities = get_llm().invoke(
            f"Extract key entities from this question as comma-separated list only:\n{question}"
        ).content.strip()
        results = []
        for entity in entities.split(","):
            entity = entity.strip()
            if not entity:
                continue
            try:
                records = graph_db.query(
                    """MATCH (n) WHERE toLower(n.id) CONTAINS toLower($e)
                       OPTIONAL MATCH (n)-[r]->(m)
                       RETURN n.id as source, type(r) as relation, m.id as target LIMIT 10""",
                    params={"e": entity}
                )
                for rec in records:
                    if rec.get("relation"):
                        results.append(f"{rec['source']} —[{rec['relation']}]→ {rec['target']}")
            except:
                pass
        return "\n".join(results) if results else "No graph data found"

    def stream_answer(prompt):
        client = GroqDirect(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    # Show messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Empty state
    if not st.session_state.documents:
        st.markdown("""
        <div class="orb-container">
            <div class="orb-wrap">
                <div class="orb-glow"></div>
                <div class="orb"></div>
            </div>
            <div class="orb-title">Ready to explore your documents?</div>
            <div class="orb-sub">Upload PDFs from the sidebar — add as many as you need.</div>
        </div>
        """, unsafe_allow_html=True)

    elif not st.session_state.active_docs:
        st.markdown("""
        <div class="orb-container">
            <div class="orb-wrap">
                <div class="orb-glow"></div>
                <div class="orb"></div>
            </div>
            <div class="orb-title">No document selected</div>
            <div class="orb-sub">Check one or more documents from the sidebar to start asking questions.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Cross-doc banner
        active_names = [st.session_state.documents[d]["name"][:20] for d in st.session_state.active_docs]
        if len(st.session_state.active_docs) > 1:
            names_str = " · ".join(active_names)
            st.markdown(f"""
            <div class="cross-doc-banner">
                <span>◈</span>
                <span>Searching across <strong>{len(st.session_state.active_docs)} documents</strong>: {names_str}</span>
            </div>
            """, unsafe_allow_html=True)

        if question := st.chat_input("Ask anything about your document(s)..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                # Collect context from ALL selected documents
                all_vector_ctx = []
                for doc_id in st.session_state.active_docs:
                    doc = st.session_state.documents[doc_id]
                    doc_chunks = doc["vector_store"].as_retriever(
                        search_kwargs={"k": 2}
                    ).invoke(question)
                    for chunk in doc_chunks:
                        all_vector_ctx.append(
                            f"[{doc['name']}]\n{chunk.page_content}"
                        )

                graph_ctx  = get_graph_context(question, st.session_state.active_docs)
                vector_ctx = "\n\n---\n\n".join(all_vector_ctx)

                doc_list = ", ".join(active_names)
                prompt = f"""You are answering questions about these documents: {doc_list}.
Use both sources below to give the best answer.
When information comes from a specific document, mention which one.
If you don't know, say "This information is not in the selected documents."

Graph relationships:
{graph_ctx}

Document text (with source labels):
{vector_ctx}

Question: {question}
Answer:"""
                answer = st.write_stream(stream_answer(prompt))

            st.session_state.messages.append({"role": "assistant", "content": answer})

# ── TAB 2: KNOWLEDGE GRAPH ────────────────
with tab_graph:
    if not st.session_state.documents:
        st.markdown("""
        <div class="orb-container">
            <div class="orb-wrap"><div class="orb-glow"></div><div class="orb"></div></div>
            <div class="orb-title">Knowledge Graph</div>
            <div class="orb-sub">Upload PDFs to visualize their entity relationships here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show graph for active docs only, or all if none selected
        display_ids = st.session_state.active_docs or list(st.session_state.documents.keys())

        # Collect all nodes/edges with doc label
        all_nodes, all_edges = [], []
        node_ids = set()
        colors = ["#8B5CF6", "#EC4899", "#F59E0B", "#10B981", "#3B82F6"]

        for i, doc_id in enumerate(display_ids):
            doc = st.session_state.documents[doc_id]
            color = colors[i % len(colors)]
            for n in doc["nodes"]:
                uid = f"{doc_id}_{n['id']}"
                if uid not in node_ids:
                    all_nodes.append({"id": uid, "raw_id": n["id"], "color": color, "doc": doc["name"]})
                    node_ids.add(uid)
            for e in doc["edges"]:
                all_edges.append({
                    "from": f"{doc_id}_{e['from']}",
                    "to": f"{doc_id}_{e['to']}",
                    "label": e["label"],
                    "color": color
                })

        # Legend
        legend_html = ""
        for i, doc_id in enumerate(display_ids):
            doc = st.session_state.documents[doc_id]
            color = colors[i % len(colors)]
            legend_html += f"""<span style="display:inline-flex;align-items:center;gap:5px;
                margin-right:12px;font-size:11px;color:#9B93B0">
                <span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block"></span>
                {doc["name"][:20]}
            </span>"""

        st.markdown(f"""
        <div class="graph-label">
            <span style="width:5px;height:5px;border-radius:50%;background:#8B5CF6;
                         box-shadow:0 0 6px #8B5CF6;display:inline-block"></span>
            {len(all_nodes)} nodes · {len(all_edges)} relations · {len(display_ids)} document(s)
        </div>
        <div style="padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.06)">
            {legend_html}
        </div>
        """, unsafe_allow_html=True)

        nodes_json = json.dumps([
            {
                "id": n["id"],
                "label": n["raw_id"][:16] + ("…" if len(n["raw_id"]) > 16 else ""),
                "title": f"{n['raw_id']}\n({n['doc']})",
                "color": {
                    "background": "#161126",
                    "border": n["color"],
                    "highlight": {"background": "#1E1833", "border": n["color"]}
                },
                "font": {"color": "#F4F0FF", "size": 11, "face": "Inter"},
                "borderWidth": 1.5,
                "size": 18
            }
            for n in all_nodes[:100]
        ])

        edges_json = json.dumps([
            {
                "from": e["from"],
                "to": e["to"],
                "label": e["label"],
                "color": {"color": e["color"] + "33", "highlight": e["color"]},
                "font": {"color": "#5C5573", "size": 9, "face": "JetBrains Mono"},
                "arrows": "to",
                "smooth": {"type": "curvedCW", "roundness": 0.25}
            }
            for e in all_edges[:150]
        ])

        graph_html = f"""
<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<style>* {{margin:0;padding:0;box-sizing:border-box;}} body {{background:#0B0912;overflow:hidden;}} #graph {{width:100%;height:500px;}}</style>
</head><body><div id="graph"></div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const options = {{
    physics: {{ enabled:true, barnesHut: {{ gravitationalConstant:-3500, springLength:130, springConstant:0.03 }}, stabilization: {{ iterations:150 }} }},
    interaction: {{ hover:true, tooltipDelay:80, zoomView:true, dragView:true }},
    layout: {{ improvedLayout:true }}
}};
new vis.Network(document.getElementById('graph'), {{nodes, edges}}, options);
</script></body></html>
"""
        components.html(graph_html, height=520)

        # Relations table
        if all_edges:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div style="font-size:11px;color:#5C5573;font-family:'JetBrains Mono',monospace;
                margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.06);
                text-transform:uppercase;letter-spacing:0.06em">Top Relations</div>""", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, edge in enumerate(all_edges[:10]):
                with cols[i % 2]:
                    from_id = edge["from"].split("_", 1)[-1]
                    to_id   = edge["to"].split("_", 1)[-1]
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                                border-radius:10px;padding:10px 14px;margin-bottom:8px;font-size:12px">
                        <span style="color:#F4F0FF">{from_id[:16]}</span>
                        <span style="color:{edge['color']};font-family:'JetBrains Mono',monospace;font-size:10px;margin:0 6px">
                            —[{edge['label']}]→
                        </span>
                        <span style="color:#F4F0FF">{to_id[:16]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    