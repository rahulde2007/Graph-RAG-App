import streamlit as st
from dotenv import load_dotenv
import os, tempfile, json

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
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
}

.stApp {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(139,92,246,0.12) 0%, var(--bg) 60%) !important;
    color: var(--text-1);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div {
    padding-top: 24px !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-3) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] label {
    color: var(--text-2) !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h2 {
    color: var(--text-1) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 1px dashed rgba(139,92,246,0.25) !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,92,246,0.5) !important;
    background: var(--accent-soft) !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 24px rgba(139,92,246,0.25), 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #7C3AED) !important;
    box-shadow: 0 0 32px rgba(139,92,246,0.35), 0 1px 3px rgba(0,0,0,0.3) !important;
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

/* ── Tabs ── */
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
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s !important;
    letter-spacing: 0.01em !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-2) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-2) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 28px 0 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
    gap: 14px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(139,92,246,0.06) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin: 6px 0 !important;
    border: 1px solid rgba(139,92,246,0.15) !important;
}
[data-testid="chatAvatarIcon-user"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #7C3AED, #A78BFA) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 16px rgba(139,92,246,0.3) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--text-1) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessage"] strong {
    color: var(--text-1) !important;
    font-weight: 500 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--border-focus) !important;
    background: rgba(255,255,255,0.06) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1), 0 0 24px rgba(139,92,246,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-1) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-3) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 12px rgba(139,92,246,0.3) !important;
}

/* ── Status + alerts ── */
[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-2) !important;
}
.stAlert { border-radius: 10px !important; }
.stSuccess {
    background: rgba(52,211,153,0.06) !important;
    border: 1px solid rgba(52,211,153,0.2) !important;
    color: var(--success) !important;
    border-radius: 10px !important;
}
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Stat pills ── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 3px 10px;
    font-size: 11px;
    color: var(--text-3);
    margin: 4px 3px 0 0;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}
.stat-pill .dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
}

/* ── Step list ── */
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    color: var(--text-3);
    font-size: 13px;
}
.step-num {
    width: 18px; height: 18px;
    border-radius: 50%;
    border: 1px solid rgba(139,92,246,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 10px;
    color: var(--accent-2);
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Orb empty state ── */
.orb-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 70px 40px 80px;
    text-align: center;
}
.orb-wrap {
    position: relative;
    width: 96px;
    height: 96px;
    margin-bottom: 28px;
}
.orb {
    width: 96px; height: 96px;
    border-radius: 50%;
    background: conic-gradient(
        from 180deg at 50% 50%,
        #7C3AED 0deg,
        #A78BFA 90deg,
        #C4B5FD 150deg,
        #7C3AED 210deg,
        #5B21B6 270deg,
        #7C3AED 360deg
    );
    box-shadow:
        0 0 40px rgba(139,92,246,0.4),
        0 0 80px rgba(139,92,246,0.15),
        inset 0 0 30px rgba(255,255,255,0.1);
    animation: orb-rotate 6s linear infinite;
}
.orb::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.25) 0%, transparent 60%);
}
.orb-glow {
    position: absolute;
    inset: -16px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,92,246,0.2) 0%, transparent 70%);
    animation: orb-pulse 3s ease-in-out infinite;
}
@keyframes orb-rotate {
    from { filter: hue-rotate(0deg); }
    to   { filter: hue-rotate(360deg); }
}
@keyframes orb-pulse {
    0%, 100% { transform: scale(1); opacity: 0.6; }
    50%       { transform: scale(1.15); opacity: 1; }
}
.orb-title {
    font-size: 22px;
    font-weight: 500;
    color: var(--text-1);
    letter-spacing: -0.02em;
    margin-bottom: 10px;
    line-height: 1.3;
}
.orb-sub {
    font-size: 14px;
    color: var(--text-3);
    max-width: 300px;
    line-height: 1.6;
}

/* ── Graph label ── */
.graph-label {
    font-size: 11px;
    color: var(--text-3);
    font-family: 'JetBrains Mono', monospace;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Mobile ── */
@media (max-width: 768px) {
    .orb-title { font-size: 18px !important; }
    [data-testid="stChatMessage"] p { font-size: 13px !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────
defaults = {
    "messages": [], "ready": False,
    "vector_store": None, "graph_db": None,
    "llm": None, "stats": {},
    "graph_nodes": [], "graph_edges": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────
with st.sidebar:
    st.markdown("## Graph RAG")
    st.markdown("&nbsp;")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and not st.session_state.ready:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Process PDF", use_container_width=True):
            with st.status("Processing...", expanded=True) as status:

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
                st.write(f"✓ {len(chunks)} chunks")

                st.write("Setting up LLM...")
                llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b")
                st.session_state.llm = llm

                st.write("Building Knowledge Graph...")
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

                st.session_state.graph_nodes = nodes
                st.session_state.graph_edges = edges
                st.session_state.graph_db = graph_db
                st.session_state.stats = {"nodes": len(nodes), "rels": len(edges), "chunks": len(chunks)}
                st.write(f"✓ {len(nodes)} nodes, {len(edges)} relations")

                st.write("Creating embeddings...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_store = Neo4jVector.from_documents(
                    documents=chunks, embedding=embeddings,
                    url=URI, username=USERNAME, password=NEO4J_PASS, database=DB_NAME,
                )
                st.session_state.vector_store = vector_store
                os.unlink(tmp_path)
                st.session_state.ready = True
                status.update(label="Ready", state="complete")

    if st.session_state.ready:
        st.success("PDF is ready")
        stats = st.session_state.stats
        st.markdown(f"""
        <div style="margin-top:8px">
            <span class="stat-pill"><span class="dot"></span>{stats.get('nodes',0)} nodes</span>
            <span class="stat-pill"><span class="dot"></span>{stats.get('rels',0)} relations</span>
            <span class="stat-pill"><span class="dot"></span>{stats.get('chunks',0)} chunks</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear & Upload New", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-item"><span class="step-num">1</span>Upload your PDF</div>
    <div class="step-item"><span class="step-num">2</span>AI builds a Knowledge Graph</div>
    <div class="step-item"><span class="step-num">3</span>Ask anything about it</div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────
st.markdown("""
<div style="padding:32px 0 18px; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:20px">
    <div style="font-size:24px;font-weight:600;color:#F4F0FF;letter-spacing:-0.03em;
                display:flex;align-items:center;gap:10px">
        <span style="width:9px;height:9px;border-radius:50%;
                     background:linear-gradient(135deg,#7C3AED,#A78BFA);
                     box-shadow:0 0 12px rgba(139,92,246,0.6);
                     display:inline-block;flex-shrink:0"></span>
        Graph RAG
    </div>
    <div style="font-size:13px;color:#5C5573;margin-top:5px;letter-spacing:0.01em">
        Document Q&amp;A powered by Knowledge Graphs
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────
tab_chat, tab_graph = st.tabs(["💬  Chat", "🕸️  Knowledge Graph"])

# ── TAB 1: CHAT ───────────────────────────
with tab_chat:

    def get_graph_context(question):
        entities = st.session_state.llm.invoke(
            f"Extract key entities from this question as comma-separated list only:\n{question}"
        ).content.strip()
        results = []
        for entity in entities.split(","):
            entity = entity.strip()
            if not entity:
                continue
            try:
                records = st.session_state.graph_db.query(
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

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.ready:
        st.markdown("""
        <div class="orb-container">
            <div class="orb-wrap">
                <div class="orb-glow"></div>
                <div class="orb"></div>
            </div>
            <div class="orb-title">Ready to explore your document?</div>
            <div class="orb-sub">Upload a PDF from the sidebar — the AI will map its knowledge graph and answer your questions.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if question := st.chat_input("Ask anything about your document..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                graph_ctx = get_graph_context(question)
                vector_ctx = "\n\n".join([
                    doc.page_content for doc in
                    st.session_state.vector_store
                        .as_retriever(search_kwargs={"k": 3})
                        .invoke(question)
                ])
                prompt = f"""Answer using both sources below.
If you don't know, say "This information is not in the document."

Graph relationships:
{graph_ctx}

Document text:
{vector_ctx}

Question: {question}
Answer:"""
                answer = st.write_stream(stream_answer(prompt))

            st.session_state.messages.append({"role": "assistant", "content": answer})

# ── TAB 2: KNOWLEDGE GRAPH ────────────────
with tab_graph:
    if not st.session_state.ready:
        st.markdown("""
        <div class="orb-container">
            <div class="orb-wrap">
                <div class="orb-glow"></div>
                <div class="orb"></div>
            </div>
            <div class="orb-title">Knowledge Graph</div>
            <div class="orb-sub">Upload a PDF to visualize its entity relationships as an interactive network.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        nodes = st.session_state.graph_nodes
        edges = st.session_state.graph_edges

        st.markdown(f"""
        <div class="graph-label">
            <span style="width:5px;height:5px;border-radius:50%;
                         background:#8B5CF6;box-shadow:0 0 6px #8B5CF6;display:inline-block"></span>
            Knowledge Graph — {len(nodes)} nodes · {len(edges)} relations
        </div>
        """, unsafe_allow_html=True)

        nodes_json = json.dumps([
            {
                "id": n["id"],
                "label": n["id"][:18] + ("…" if len(n["id"]) > 18 else ""),
                "title": n["id"],
                "color": {
                    "background": "#161126",
                    "border": "#7C3AED",
                    "highlight": {"background": "#1E1833", "border": "#A78BFA"}
                },
                "font": {"color": "#F4F0FF", "size": 11, "face": "Inter"},
                "borderWidth": 1.5,
                "size": 18
            }
            for n in nodes[:80]
        ])

        edges_json = json.dumps([
            {
                "from": e["from"],
                "to": e["to"],
                "label": e["label"],
                "color": {"color": "rgba(255,255,255,0.06)", "highlight": "#8B5CF6"},
                "font": {"color": "#5C5573", "size": 9, "face": "JetBrains Mono"},
                "arrows": "to",
                "smooth": {"type": "curvedCW", "roundness": 0.25}
            }
            for e in edges[:120]
        ])

        graph_html = f"""
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0B0912; overflow:hidden; }}
#graph {{ width:100%; height:520px; }}
</style>
</head>
<body>
<div id="graph"></div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const container = document.getElementById('graph');
const options = {{
    physics: {{
        enabled: true,
        barnesHut: {{
            gravitationalConstant: -3500,
            springLength: 130,
            springConstant: 0.03
        }},
        stabilization: {{ iterations: 150 }}
    }},
    interaction: {{
        hover: true,
        tooltipDelay: 80,
        zoomView: true,
        dragView: true
    }},
    layout: {{ improvedLayout: true }}
}};
new vis.Network(container, {{ nodes, edges }}, options);
</script>
</body>
</html>
"""
        components.html(graph_html, height=530)

        if edges:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:11px;color:#5C5573;font-family:'JetBrains Mono',monospace;
                        margin-bottom:14px;padding-bottom:10px;
                        border-bottom:1px solid rgba(255,255,255,0.06);
                        text-transform:uppercase;letter-spacing:0.06em">
                Top Relations
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(2)
            for i, edge in enumerate(edges[:10]):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02);
                                border:1px solid rgba(255,255,255,0.06);
                                border-radius:10px;
                                padding:10px 14px;
                                margin-bottom:8px;
                                font-size:12px;
                                backdrop-filter:blur(10px)">
                        <span style="color:#F4F0FF">{edge['from'][:18]}</span>
                        <span style="color:#7C3AED;font-family:'JetBrains Mono',monospace;
                                     font-size:10px;margin:0 6px">
                            —[{edge['label']}]→
                        </span>
                        <span style="color:#F4F0FF">{edge['to'][:18]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    