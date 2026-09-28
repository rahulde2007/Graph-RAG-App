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

# ── Page Config ───────────────────────────
st.set_page_config(
    page_title="Graph RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:    #0C0E16;
    --bg-secondary:  #13151F;
    --bg-card:       #1A1D2E;
    --bg-hover:      #21243A;
    --accent:        #6C63FF;
    --accent-glow:   rgba(108,99,255,0.15);
    --accent-soft:   rgba(108,99,255,0.08);
    --text-primary:  #EEEEF5;
    --text-secondary:#8B8FA8;
    --text-muted:    #555870;
    --border:        #252838;
    --border-accent: rgba(108,99,255,0.4);
    --success:       #4ADE80;
    --success-soft:  rgba(74,222,128,0.1);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-primary); color: var(--text-primary); }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }

[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: var(--text-secondary) !important; font-size: 13px; }
[data-testid="stSidebar"] h2 {
    color: var(--text-primary) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 1.5px dashed var(--border);
    border-radius: 10px;
    padding: 4px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--border-accent);
    background: var(--accent-soft);
}

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}
.stButton > button:hover {
    background: #7B73FF !important;
    box-shadow: 0 0 30px rgba(108,99,255,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    box-shadow: none !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
    transform: none !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 0;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--accent-soft) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 24px 0 !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 8px 0 !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--accent-soft) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 8px 0 !important;
    border: 1px solid var(--border-accent) !important;
}
[data-testid="chatAvatarIcon-user"] { background: var(--bg-hover) !important; }
[data-testid="chatAvatarIcon-assistant"] {
    background: var(--accent) !important;
    box-shadow: 0 0 10px var(--accent-glow) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--text-primary) !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
}

[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--border-accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stChatInputSubmitButton"] button {
    background: var(--accent) !important;
    border-radius: 8px !important;
}

[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stAlert { border-radius: 8px !important; }
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--text-secondary);
    margin: 4px 4px 0 0;
    font-family: 'JetBrains Mono', monospace;
}
.stat-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
}

.step-item {
    display: flex; align-items: flex-start;
    gap: 10px; padding: 8px 0;
    color: var(--text-secondary); font-size: 13px;
}

/* Always show sidebar — hide collapse button */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    min-width: 260px !important;
    max-width: 260px !important;
    transform: none !important;
    visibility: visible !important;
}

.step-num {
    width: 20px; height: 20px;
    border-radius: 50%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: var(--accent); flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}

.empty-state {
    text-align: center;
    padding: 80px 40px;
    color: var(--text-muted);
}
.empty-state .icon { font-size: 48px; margin-bottom: 16px; opacity: 0.4; }
.empty-state p { font-size: 15px; max-width: 320px; margin: 0 auto; line-height: 1.6; }

.graph-label {
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}

@media (max-width: 768px) {
    [data-testid="stChatMessage"] p { font-size: 14px !important; }
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
    st.markdown("## Upload PDF")
    uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

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
                        edges.append({
                            "from": r.source.id,
                            "to": r.target.id,
                            "label": r.type
                        })

                st.session_state.graph_nodes = nodes
                st.session_state.graph_edges = edges
                st.session_state.graph_db = graph_db
                st.session_state.stats = {
                    "nodes": len(nodes),
                    "rels": len(edges),
                    "chunks": len(chunks)
                }
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
<div style="padding:36px 0 20px; border-bottom:1px solid #252838; margin-bottom:24px">
    <div style="font-size:26px;font-weight:600;color:#EEEEF5;letter-spacing:-0.02em;display:flex;align-items:center;gap:10px">
        <span style="width:8px;height:8px;border-radius:50%;background:#6C63FF;box-shadow:0 0 10px #6C63FF;display:inline-block"></span>
        Graph RAG
    </div>
    <div style="font-size:14px;color:#555870;margin-top:6px">Document Q&A powered by Knowledge Graphs</div>
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

    # Show messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.ready:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">◈</div>
            <p>Upload a PDF from the sidebar — the AI will map its knowledge and answer your questions.</p>
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
        <div class="empty-state">
            <div class="icon">◈</div>
            <p>Upload a PDF to see its Knowledge Graph visualized here.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        nodes = st.session_state.graph_nodes
        edges = st.session_state.graph_edges

        st.markdown(f"""
        <div class="graph-label">
            <span style="width:6px;height:6px;border-radius:50%;background:#6C63FF;display:inline-block"></span>
            Knowledge Graph — {len(nodes)} nodes · {len(edges)} relations
        </div>
        """, unsafe_allow_html=True)

        nodes_json = json.dumps([
            {
                "id": n["id"],
                "label": n["id"][:20] + ("..." if len(n["id"]) > 20 else ""),
                "title": n["id"],
                "color": {
                    "background": "#1A1D2E",
                    "border": "#6C63FF",
                    "highlight": {"background": "#21243A", "border": "#7B73FF"}
                },
                "font": {"color": "#EEEEF5", "size": 12, "face": "Inter"},
                "borderWidth": 2,
                "size": 20
            }
            for n in nodes[:80]
        ])

        edges_json = json.dumps([
            {
                "from": e["from"],
                "to": e["to"],
                "label": e["label"],
                "color": {"color": "#252838", "highlight": "#6C63FF"},
                "font": {"color": "#555870", "size": 10, "face": "JetBrains Mono"},
                "arrows": "to",
                "smooth": {"type": "curvedCW", "roundness": 0.2}
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
body {{ background:#0C0E16; overflow:hidden; }}
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
            gravitationalConstant: -3000,
            springLength: 120,
            springConstant: 0.04
        }},
        stabilization: {{ iterations: 150 }}
    }},
    interaction: {{
        hover: true,
        tooltipDelay: 100,
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
            <div style="font-size:13px;color:#555870;font-family:'JetBrains Mono',monospace;
                        margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #252838">
                Top Relations
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(2)
            for i, edge in enumerate(edges[:10]):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style="background:#1A1D2E;border:1px solid #252838;border-radius:8px;
                                padding:10px 14px;margin-bottom:8px;font-size:13px">
                        <span style="color:#EEEEF5">{edge['from'][:20]}</span>
                        <span style="color:#6C63FF;font-family:'JetBrains Mono',monospace;
                                     font-size:11px;margin:0 6px">—[{edge['label']}]→</span>
                        <span style="color:#EEEEF5">{edge['to'][:20]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    