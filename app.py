import streamlit as st
import os
import time
import base64
import json
import fitz  # PyMuPDF
from groq import Groq
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="GraphRAG Elite", page_icon="🕸️", layout="wide")

# --- 1. SECRETS MANAGEMENT ---
# Fetches from Hugging Face Secrets (Environment Variables)
GROQ_KEY = os.environ.get("GROQ_API_KEY")
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD")

# Sidebar Fallback (for local testing)
with st.sidebar:
    st.header("⚙️ Configuration")
    if not GROQ_KEY:
        GROQ_KEY = st.text_input("Groq API Key", type="password")
    if not NEO4J_URI:
        NEO4J_URI = st.text_input("Neo4j URI")
    if not NEO4J_PASS:
        NEO4J_PASS = st.text_input("Neo4j Password", type="password")
        
    st.divider()
    st.info("System: Llama-4-Scout (Vision) + Cross-Encoder (Reranking)")

# --- 2. BACKEND LOGIC ---
@st.cache_resource
def load_models():
    # Load models once and cache them
    embed = SentenceTransformer('all-MiniLM-L6-v2')
    rerank = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embed, rerank

class GraphRAG:
    def __init__(self, g_key, n_uri, n_pass):
        self.driver = GraphDatabase.driver(n_uri, auth=("neo4j", n_pass))
        self.groq = Groq(api_key=g_key)
        self.embed, self.rerank = load_models()
        self.index = "chunk_vector_index"
        
        # Ensure Schema Exists
        try:
            with self.driver.session() as session:
                session.run(f"CREATE VECTOR INDEX `{self.index}` IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) OPTIONS {{ indexConfig: {{ `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' }} }}")
        except Exception as e:
            st.error(f"Database Connection Failed: {e}")

    def ingest_pdf(self, uploaded_file):
        """Reads PDF -> Extracts Text/Vision -> Pushes to Neo4j"""
        status = st.empty()
        progress_bar = st.progress(0)
        status.info("📄 Processing PDF...")
        
        # 1. Save temp file safely
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        doc = fitz.open("temp.pdf")
        total_pages = len(doc)
        count = 0
        
        for i, page in enumerate(doc):
            text = page.get_text()
            
            # Vision Logic: Check for charts
            if len(page.get_images()) > 0:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                b64_img = base64.b64encode(img_bytes).decode('utf-8')
                
                # Ask Llama Vision
                try:
                    resp = self.groq.chat.completions.create(
                        messages=[{"role":"user", "content":[{"type":"text", "text":"Extract CHARTS/TABLES to text."}, {"type":"image_url", "image_url":{"url":f"data:image/jpeg;base64,{b64_img}"}}]}],
                        model="meta-llama/llama-4-scout-17b-16e-instruct"
                    )
                    text += f"\n[CHART DATA]: {resp.choices[0].message.content}"
                except: pass
            
            # Ingest if valid
            if len(text) > 50:
                vec = self.embed.encode(text).tolist()
                
                # Graph Extraction
                ents = []
                try:
                    g_resp = self.groq.chat.completions.create(
                        messages=[{"role":"user", "content":f"Extract entities JSON from:\n{text[:2000]}"}],
                        model="llama-3.3-70b-versatile",
                        response_format={"type":"json_object"}
                    )
                    ents = json.loads(g_resp.choices[0].message.content).get('entities', [])
                    # Filter bad entities
                    ents = [e for e in ents if e.get('source') and e.get('target')]
                except Exception as e:
                    print(f"Extraction failed on page {i}: {e}")
                    ents = [] # Continue even if extraction fails

                # Cypher Write
                query = """
                MERGE (c:Chunk {textId: $h}) ON CREATE SET c.text=$t, c.embedding=$v
                WITH c UNWIND $e as ent
                MERGE (s:Entity {name: ent.source}) MERGE (t:Entity {name: ent.target})
                MERGE (s)-[:RELATED {type: ent.relation}]->(t)
                MERGE (c)-[:MENTIONS]->(s) MERGE (c)-[:MENTIONS]->(t)
                """
                with self.driver.session() as session:
                    session.run(query, h=str(hash(text)), t=text, v=vec, e=ents)
                
                count += 1
                status.text(f"✅ Ingested Page {i+1} ({len(ents)} entities found)...")
                progress_bar.progress((i + 1) / total_pages)
                
        status.success(f"🎉 Processed {count} pages successfully!")
        doc.close() # Close file handle
        os.remove("temp.pdf") # Clean up

    def query(self, text):
        # 1. Retrieval
        vec = self.embed.encode(text).tolist()
        
        # CRITICAL FIX: Changed MATCH to OPTIONAL MATCH
        # This ensures we get results even if the Graph Extraction failed for that chunk
        q = f"""
        CALL db.index.vector.queryNodes('{self.index}', 10, $v) 
        YIELD node AS c, score 
        OPTIONAL MATCH (c)-[:MENTIONS]->(e) 
        RETURN c.text as text, collect(distinct e.name) as ents
        """
        
        with self.driver.session() as session:
            docs = [dict(r) for r in session.run(q, v=vec)]
            
        # 2. Reranking
        if not docs: return "No info found.", []
        
        # Deduplicate docs based on text to avoid repeats
        unique_docs = {d['text']: d for d in docs}.values()
        docs = list(unique_docs)

        pairs = [[text, d['text']] for d in docs]
        scores = self.rerank.predict(pairs)
        top_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:3]
        
        # 3. Generation
        context = "\n".join([d[0]['text'] for d in top_docs])[:15000]
        
        prompt = f"""
        Use the context below to answer the question.
        
        Context:
        {context}
        
        Question: {text}
        Answer:
        """
        
        try:
            ans = self.groq.chat.completions.create(
                messages=[{"role":"user", "content":prompt}],
                model="llama-3.1-8b-instant"
            ).choices[0].message.content
        except Exception as e:
            ans = f"LLM Error: {e}"
            
        return ans, top_docs

# --- 3. MAIN UI ---
st.title("🕸️ GraphRAG Elite")

# Check keys before proceeding
if not GROQ_KEY or not NEO4J_URI:
    st.warning("⚠️ API Keys missing! Add them in Settings > Secrets or the Sidebar.")
    st.stop()

# Initialize App
try:
    rag = GraphRAG(GROQ_KEY, NEO4J_URI, NEO4J_PASS)
except Exception as e:
    st.error(f"Failed to connect to Neo4j: {e}")
    st.stop()

# Upload Section
with st.expander("📂 Upload Knowledge (PDF)", expanded=False):
    up_file = st.file_uploader("Upload a financial report", type="pdf")
    if up_file and st.button("Ingest File"):
        rag.ingest_pdf(up_file)

# Chat Section
if "messages" not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.spinner("🕷️ Thinking..."):
        ans, evidence = rag.query(prompt)
        st.chat_message("assistant").markdown(ans)
        
        with st.expander("🔍 View Evidence"):
            for doc, score in evidence:
                st.caption(f"Relevance Score: {score:.2f}")
                st.info(doc['text'][:300] + "...")
                # Show entities if they exist
                if doc['ents']:
                    st.caption(f"Entities: {', '.join(doc['ents'][:5])}")
                
        st.session_state.messages.append({"role":"assistant", "content":ans})